import time
import random
import math

def _score_mask(M):
    """
    Calculate the number of distinct d > 0 such that there exists a
    with a, a+d, a+2d in the set represented by bitmask M.
    M is an integer where bit i is 1 if i is in the set.
    Indices 1..512 are used.
    """
    count = 0
    for d in range(1, 256):
        if (M & (M >> d) & (M >> 2 * d)) != 0:
            count += 1
    return count

def _score_set(s):
    """Score a set of integers by counting distinct 3-term AP differences."""
    M = 0
    for x in s:
        M |= (1 << x)
    return _score_mask(M)

def _can_realize_diff(d, S):
    """Check if difference d can be realized by some 3-term AP in set S."""
    S_set = set(S)
    for a in S:
        if (a + d) in S_set and (a + 2*d) in S_set:
            return True
    return False

def _build_set_for_diffs(target_diffs, N=32, MAXV=512, time_budget=30):
    """
    Build a set of N integers that realizes as many target differences as possible.
    Uses greedy construction + local search.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Initialize with elements that help realize multiple target diffs
    # Greedy: pick elements that maximize coverage of target diffs
    S = set()
    remaining_diffs = set(target_diffs)
    
    # First, pick a base element
    base = random.randint(1, MAXV // 3)
    S.add(base)
    
    # Greedily add elements
    while len(S) < N:
        if time.time() > deadline - 5:
            break
        best_elem = None
        best_gain = -1
        best_covered = set()
        
        # Try candidate positions
        candidates = list(range(1, MAXV + 1))
        random.shuffle(candidates)
        
        for cand in candidates:
            if cand in S:
                continue
            if time.time() > deadline - 5:
                break
            
            new_S = S | {cand}
            # Count how many target diffs are now realized
            covered = set()
            for d in remaining_diffs:
                if _can_realize_diff(d, new_S):
                    covered.add(d)
            
            gain = len(covered)
            if gain > best_gain:
                best_gain = gain
                best_elem = cand
                best_covered = covered
        
        if best_elem is not None:
            S.add(best_elem)
            remaining_diffs -= best_covered
    
    S = list(S)
    if len(S) < N:
        # Fill with random elements
        while len(S) < N:
            r = random.randint(1, MAXV)
            if r not in S:
                S.append(r)
    
    cur_score = _score_set(S)
    best_S = S[:]
    best_score = cur_score
    
    # Local search: swap elements to improve score
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_found = False
        best_new_S = None
        best_new_score = cur_score
        
        for i in range(N):
            if time.time() > deadline:
                break
            rem = S[i]
            for add in range(1, MAXV + 1):
                if add in S:
                    continue
                new_S = S[:]
                new_S[i] = add
                new_score = _score_set(new_S)
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_new_S = new_S
                    best_found = True
        
        if best_found:
            S = best_new_S
            cur_score = best_new_score
            improved = True
            if cur_score > best_score:
                best_score = cur_score
                best_S = S[:]
    
    return best_S, best_score

def _find_dense_diff_cluster(MAXV=512, N=32, time_budget=20):
    """
    Find a dense cluster of differences that can be simultaneously realized.
    Two differences d1, d2 are 'compatible' if there exists a set of size N
    that realizes both. We approximate this by checking if they can share
    a common base or be realized by overlapping APs.
    
    Strategy: Find differences that have many overlapping AP representations.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # For each difference d, count how many 3-term APs (a, a+d, a+2d) exist in [1, MAXV]
    # This gives us the "capacity" of each difference
    diff_capacity = {}
    for d in range(1, 256):
        # a can range from 1 to MAXV - 2*d
        max_a = MAXV - 2 * d
        if max_a >= 1:
            diff_capacity[d] = max_a
        else:
            diff_capacity[d] = 0
    
    # Sort differences by capacity (higher capacity = more flexible)
    sorted_diffs = sorted(diff_capacity.keys(), key=lambda d: diff_capacity[d], reverse=True)
    
    # Greedily build a cluster: start with high-capacity diffs, add compatible ones
    # Two diffs d1, d2 are compatible if they can be realized by a set of size N
    # Approximation: they are compatible if there exists an a such that
    # {a, a+d1, a+2*d1} and {a, a+d2, a+2*d2} share at least one element
    # This is always true if we choose a appropriately, so use a different criterion.
    
    # Better criterion: diffs are "clusterable" if they have similar magnitudes
    # or if their APs can overlap significantly.
    
    # Use a simpler approach: pick diffs that are multiples of a common base
    # or that are in arithmetic progression themselves.
    
    # Strategy: Find a base difference b, then take {b, 2b, 3b, ...} up to some limit
    # These can be realized by a set like {1, 1+b, 1+2b, 1+3b, ...}
    
    best_cluster = []
    best_cluster_size = 0
    
    for base in range(1, 30):
        if time.time() > deadline:
            break
        
        cluster = []
        for d in range(base, 256, base):
            if diff_capacity[d] >= N:  # Can fit N elements
                cluster.append(d)
        
        if len(cluster) > best_cluster_size:
            best_cluster_size = len(cluster)
            best_cluster = cluster[:]
    
    # Also try clusters based on consecutive differences
    for start in range(1, 50):
        if time.time() > deadline:
            break
        cluster = []
        for d in range(start, min(start + 30, 256)):
            if diff_capacity[d] >= N:
                cluster.append(d)
        if len(cluster) > best_cluster_size:
            best_cluster_size = len(cluster)
            best_cluster = cluster[:]
    
    # Also try random clusters
    for _ in range(100):
        if time.time() > deadline:
            break
        # Pick a random center and radius
        center = random.randint(10, 100)
        radius = random.randint(5, 20)
        cluster = [d for d in range(max(1, center - radius), min(256, center + radius + 1))
                   if diff_capacity[d] >= N]
        if len(cluster) > best_cluster_size:
            best_cluster_size = len(cluster)
            best_cluster = cluster[:]
    
    return best_cluster

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    N = 32
    MAXV = 512
    
    # Phase 1: Find a dense difference cluster
    cluster = _find_dense_diff_cluster(MAXV, N, time_budget=15)
    
    # Phase 2: Build sets targeting different subsets of the cluster
    best_S = None
    best_score = 0
    
    # Try multiple random subsets of the cluster
    num_trials = 5
    time_per_trial = (deadline - start_time - 10) / num_trials
    
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
        
        # Pick a random subset of the cluster (or the whole cluster if small)
        if len(cluster) <= 20:
            target_diffs = cluster[:]
        else:
            target_diffs = random.sample(cluster, min(20, len(cluster)))
        
        S, score = _build_set_for_diffs(target_diffs, N, MAXV, time_budget=time_per_trial)
        
        if score > best_score:
            best_score = score
            best_S = S[:]
    
    # Phase 3: Final polish with local search
    if best_S is not None and time.time() < deadline - 5:
        polish_time = deadline - time.time() - 5
        S = best_S[:]
        cur_score = best_score
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                rem = S[i]
                for add in range(1, MAXV + 1):
                    if add in S:
                        continue
                    new_S = S[:]
                    new_S[i] = add
                    new_score = _score_set(new_S)
                    if new_score > cur_score:
                        S = new_S
                        cur_score = new_score
                        improved = True
                        break
        
        if cur_score > best_score:
            best_score = cur_score
            best_S = S[:]
    
    if best_S is None:
        # Fallback: random set
        best_S = random.sample(range(1, MAXV + 1), N)
    
    return best_S
