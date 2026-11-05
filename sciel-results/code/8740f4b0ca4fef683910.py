import time
from itertools import combinations

def is_sidon(subset):
    """Check if a given subset has all unique pair sums including diagonals."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(candidate_set):
    """Find the size of the largest Sidon subset of the given candidate set."""
    n = len(candidate_set)
    # Check from largest possible subset size down
    for k in range(n, 0, -1):
        for subset in combinations(candidate_set, k):
            if is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85 seconds to be safe (90s limit)
    
    best_set = None
    best_size = 12
    
    # Try structured candidates with many additive collisions
    # Clusters of consecutive numbers create many sum collisions
    structured_candidates = [
        # 3 clusters of 4 consecutive
        [0,1,2,3, 20,21,22,23, 40,41,42,43],
        # 4 clusters of 3 consecutive
        [0,1,2, 30,31,32, 60,61,62, 90,91,92],
        # 6 clusters of 2 consecutive
        [0,1, 20,21, 40,41, 60,61, 80,81, 100,101],
        # Mix: some clusters
        [0,1,2,3, 15,16,17, 30,31,32,33, 50,51],
        # Arithmetic progression with small step
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        # Dense at start
        [0,1,2,3,4,5,6,7,8,9,10,11],
        # Sparse but with collisions
        [0,1,2, 10,11,12, 20,21,22, 30,31,32],
    ]
    
    for c in structured_candidates:
        if len(set(c)) != 12:
            continue
        if any(x < 0 or x > 128 for x in c):
            continue
        sz = max_sidon_size(c)
        if best_set is None or sz < best_size:
            best_size = sz
            best_set = c
        if best_size <= 4:
            break
    
    # Random search with bias toward clustered sets
    import random
    random.seed(42)
    
    iterations = 0
    while time.time() < deadline:
        # Generate random set, sometimes clustered
        if iterations % 3 == 0:
            # Clustered: pick 3-4 cluster centers, put 3-4 elements near each
            num_clusters = random.choice([3, 4, 6])
            centers = random.sample(range(5, 125), num_clusters)
            s = set()
            while len(s) < 12:
                center = random.choice(centers)
                offset = random.randint(-2, 2)
                val = center + offset
                if 0 <= val <= 128:
                    s.add(val)
            s = list(s)
            if len(s) != 12:
                continue
        elif iterations % 3 == 1:
            # Semi-random: mix of close and far
            s = set()
            while len(s) < 12:
                if s:
                    base = random.choice(list(s))
                    val = base + random.randint(-3, 3)
                else:
                    val = random.randint(0, 128)
                if 0 <= val <= 128:
                    s.add(val)
            s = list(s)
            if len(s) != 12:
                continue
        else:
            # Fully random
            s = random.sample(range(129), 12)
        
        sz = max_sidon_size(s)
        if sz < best_size:
            best_size = sz
            best_set = s
            if best_size <= 4:
                break
        iterations += 1
    
    return best_set
