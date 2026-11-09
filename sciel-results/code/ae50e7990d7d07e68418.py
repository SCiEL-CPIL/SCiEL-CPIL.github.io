import itertools
import random
import time

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

def idx(p):
    return p[0] * N + p[1]

def _precompute_blocks():
    blocks = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        pi = POINTS[i]
        for j in range(i + 1, NUM_POINTS):
            pj = POINTS[j]
            d2 = (pi[0] - pj[0])**2 + (pi[1] - pj[1])**2
            mask = 0
            for k in range(NUM_POINTS):
                pk = POINTS[k]
                d_ik = (pi[0] - pk[0])**2 + (pi[1] - pk[1])**2
                d_jk = (pj[0] - pk[0])**2 + (pj[1] - pk[1])**2
                if d2 == d_ik or d2 == d_jk or d_ik == d_jk:
                    mask |= (1 << k)
            blocks[i][j] = mask
            blocks[j][i] = mask
    return blocks

BLOCKS = _precompute_blocks()

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_set = None
    best_size = float('inf')
    
    indices = list(range(NUM_POINTS))
    
    for _ in range(100): # More restarts
        if time.time() > deadline:
            break
            
        random.shuffle(indices)
        
        S = []
        U_S = 0 # Union of blocks for pairs in S
        
        for p_idx in indices:
            # Check if p_idx can be added
            # It can be added if it is not in U_S
            if (1 << p_idx) & U_S:
                continue
                
            # Add p_idx to S
            # Update U_S
            # New pairs are (p_idx, q) for all q in S
            for q_idx in S:
                U_S |= BLOCKS[p_idx][q_idx]
            S.append(p_idx)
            
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for p_idx in list(S):
                # Check if p_idx can be removed
                # p_idx can be removed if it is blocked by S \ {p_idx}
                # i.e., p_idx is in the intersection of blocks for all pairs in S \ {p_idx}
                
                # If S has size <= 2, S \ {p_idx} has < 2 points, so no pairs.
                # Intersection over empty set is all points?
                # If there are no pairs, any point is "blocked"?
                # No, if there are no pairs, no isosceles can be formed.
                # So any point can be added.
                # So S \ {p_idx} is NOT maximal if it has size < 3?
                # Wait. If S has size 2, S \ {p_idx} has size 1.
                # A set of size 1 is no-isosceles.
                # Is it maximal?
                # If we add a point, we get size 2. No isosceles.
                # So size 1 is not maximal.
                # So we cannot remove points from a set of size 2 if it makes it size 1?
                # Actually, the problem implies N=16, so minimal maximal sets are likely larger.
                
                # For S \ {p_idx} to be maximal, every point not in S \ {p_idx} must be blocked.
                # If S \ {p_idx} has < 2 points, it is not maximal (unless N is very small).
                # So we should only try to remove if len(S) >= 3.
                
                if len(S) < 3:
                    continue
                
                can_remove = True
                S_copy = S[:]
                S_copy.remove(p_idx)
                
                # Check intersection
                # Initialize with all 1s?
                # We need to check if p_idx is in blocks[a][b] for all (a,b) in S_copy
                for i in range(len(S_copy)):
                    for j in range(i + 1, len(S_copy)):
                        a_idx = S_copy[i]
                        b_idx = S_copy[j]
                        if not ((1 << p_idx) & BLOCKS[a_idx][b_idx]):
                            can_remove = False
                            break
                    if not can_remove:
                        break
                
                if can_remove:
                    S.remove(p_idx)
                    improved = True
                    break
        
        current_size = len(S)
        if current_size < best_size:
            best_size = current_size
            best_set = S[:]
            
    return [POINTS[i] for i in best_set]
