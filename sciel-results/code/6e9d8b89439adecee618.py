import itertools
import random
import time
import math

N = 16
NUM_POINTS = N * N

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    idx_to_point = {i: p for p, i in point_to_idx.items()}
    
    # Precompute bad_pair[i][j] as a bitset of points k such that (i, j, k) is isosceles.
    bad_pair = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    
    # Precompute distances
    dist = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        pi = idx_to_point[i]
        for j in range(i + 1, NUM_POINTS):
            pj = idx_to_point[j]
            d = _d2(pi, pj)
            dist[i][j] = d
            dist[j][i] = d
            
    for i in range(NUM_POINTS):
        for j in range(i + 1, NUM_POINTS):
            mask = 0
            d_ij = dist[i][j]
            for k in range(NUM_POINTS):
                if k == i or k == j:
                    continue
                d_ik = dist[i][k]
                d_jk = dist[j][k]
                if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                    mask |= (1 << k)
            bad_pair[i][j] = mask
            bad_pair[j][i] = mask
            
    all_mask = (1 << NUM_POINTS) - 1
    
    best_set = None
    best_size = float('inf')
    
    def _construct_greedy(seed):
        random.seed(seed)
        S = []
        S_mask = 0
        blocked = 0
        
        while True:
            available = all_mask & ~S_mask & ~blocked
            if available == 0:
                break
                
            # Get list of available indices
            avail_list = []
            temp = available
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                avail_list.append(idx)
                temp ^= lsb
                
            if not avail_list:
                break
                
            best_p = None
            best_gain = -1
            
            # Sample candidates to speed up
            if len(avail_list) > 50:
                candidates = random.sample(avail_list, 50)
            else:
                candidates = avail_list
                
            for p in candidates:
                new_blocked = 0
                for r in S:
                    new_blocked |= bad_pair[p][r]
                # Count how many currently available points are blocked
                gain = (new_blocked & available).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    
            if best_p is None:
                best_p = random.choice(avail_list)
                
            S.append(best_p)
            S_mask |= (1 << best_p)
            
            # Update blocked
            for r in S[:-1]:
                blocked |= bad_pair[best_p][r]
                
        return S
        
    def _is_maximal(S):
        if not S:
            return True
        S_mask = 0
        for p in S:
            S_mask |= (1 << p)
            
        blocked = 0
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                blocked |= bad_pair[S[i]][S[j]]
                
        available = all_mask & ~S_mask & ~blocked
        return available == 0
        
    def _prune(S):
        S_list = S[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(S_list)):
                p = S_list[i]
                S_temp = S_list[:i] + S_list[i+1:]
                if _is_maximal(S_temp):
                    S_list = S_temp
                    improved = True
                    break
        return S_list

    num_trials = 200
    for t in range(num_trials):
        if time.time() > deadline - 10:
            break
        S = _construct_greedy(t)
        S = _prune(S)
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
    if best_set is None:
        return []
        
    return [idx_to_point[p] for p in best_set]
