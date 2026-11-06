"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import itertools

# Precompute all grid points
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute squared distances between all pairs of points
# dist2[i][j] = squared distance between POINTS[i] and POINTS[j]
dist2 = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2
        dist2[i][j] = d
        dist2[j][i] = d

def is_isosceles(i, j, k):
    """Check if points i, j, k form an isosceles triangle."""
    d1 = dist2[i][j]
    d2 = dist2[i][k]
    d3 = dist2[j][k]
    return d1 == d2 or d1 == d3 or d2 == d3

def is_valid_set(set_indices):
    """Check if the set of point indices contains no isosceles triangles."""
    # Optimization: If we maintain a list of points, we can check pairs.
    # For SA, we need fast validity checks for single moves.
    pass

def can_add_point(set_indices, p_idx):
    """Check if point p_idx can be added to set_indices without creating an isosceles triangle."""
    # An isosceles triangle is formed if p_idx and any two existing points a, b
    # satisfy dist(p,a) == dist(p,b) or dist(p,a) == dist(a,b) or dist(p,b) == dist(a,b).
    # Actually, simpler: For any pair (a, b) in set_indices, if (p, a, b) is isosceles, we cannot add p.
    # This is O(k^2) where k is size of set.
    
    # Faster check:
    # 1. For each a in set_indices, check if dist(p, a) equals dist(b, c) for any b,c? No, that's not right.
    # The condition is: No three points in the NEW set form an isosceles triangle.
    # Since the OLD set is valid, we only need to check triangles involving the new point p.
    # Triangles involving p are (p, a, b) where a, b are in the OLD set.
    # (p, a, b) is isosceles if:
    #   dist(p, a) == dist(p, b)  OR
    #   dist(p, a) == dist(a, b)  OR
    #   dist(p, b) == dist(a, b)
    
    # Let's precompute for each point p, the set of points q such that dist(p,q) is unique? No.
    # Let's just iterate over pairs in the current set.
    # If set size is ~50, pairs are ~1250. This is fast enough for SA steps if we do it smartly.
    
    # Alternative: Precompute for each point p, and each distance d, the list of points at distance d from p.
    # If we add p, we create a conflict if:
    # 1. There exist a, b in S such that dist(p,a) == dist(p,b).
    #    This means p sees two points in S at the same distance.
    # 2. There exist a, b in S such that dist(p,a) == dist(a,b).
    #    This means for some a in S, dist(p,a) is equal to the distance between a and some other point b in S.
    
    # Let's use a simpler O(k^2) check for now, but optimized.
    s_list = list(set_indices)
    k = len(s_list)
    if k < 2:
        return True
        
    # Check condition 1: dist(p, a) == dist(p, b)
    # Group points in S by their distance to p. If any group has size > 1, conflict.
    dist_to_p = {}
    for a in s_list:
        d = dist2[p_idx][a]
        if d in dist_to_p:
            return False # Two points at same distance from p -> isosceles (p, a, b)
        dist_to_p[d] = a
        
    # Check condition 2 & 3: dist(p, a) == dist(a, b) or dist(p, b) == dist(a, b)
    # This is equivalent to: For any pair (a, b) in S, if dist(p, a) == dist(a, b) OR dist(p, b) == dist(a, b), conflict.
    # We can precompute for each point a, a map of distance -> count of points at that distance from a?
    # No, we need to know if dist(p, a) matches dist(a, b) for a specific b.
    
    # Let's just iterate pairs (a, b) in S.
    # For each pair (a, b):
    #   d_ab = dist2[a][b]
    #   if dist2[p_idx][a] == d_ab: return False
    #   if dist2[p_idx][b] == d_ab: return False
    
    # This is O(k^2). With k=50, 2500 ops. Very fast.
    for i in range(k):
        a = s_list[i]
        for j in range(i + 1, k):
            b = s_list[j]
            d_ab = dist2[a][b]
            if dist2[p_idx][a] == d_ab or dist2[p_idx][b] == d_ab:
                return False
    return True

def is_maximal(set_indices):
    """Check if no point can be added to the set."""
    s_set = set(set_indices)
    for p_idx in range(NUM_POINTS):
        if p_idx in s_set:
            continue
        if can_add_point(set_indices, p_idx):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Initial solution: Greedy construction
    # We use a random order to get diverse starting points
    random.seed(42)
    
    best_set = None
    best_size = NUM_POINTS + 1
    
    # Run multiple restarts with different random seeds/orders
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Generate a random order of points
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        current_set = []
        for p_idx in order:
            if can_add_point(current_set, p_idx):
                current_set.append(p_idx)
        
        # Local search: Simulated Annealing
        # State: set of points
        # Energy: -len(set) (we want to minimize len, so maximize -len)
        # Or just minimize len, with penalties for invalidity/non-maximality.
        
        # Let's use a penalty method.
        # Energy = len(set) + M1 * (num_invalid_triangles) + M2 * (1 if not maximal else 0)
        # But checking num_invalid_triangles is expensive.
        # Instead, let's stick to valid states only, and use "Add/Remove" moves.
        # If we remove a point, the set might become non-maximal.
        # If we add a point, the set might become invalid.
        
        # Strategy: 
        # 1. Start with a valid maximal set (from greedy).
        # 2. Try to remove points. If removal keeps it valid (always true) and maximal (check), accept if it reduces size.
        #    If removal makes it non-maximal, we might accept it with probability based on "how non-maximal" it is?
        #    Or, we can try to remove a point, and if it becomes non-maximal, immediately try to add the best point to restore maximality?
        
        # Let's try a simpler "Iterative Improvement" with Randomized Restart and Randomized Removal Order.
        # 1. Greedy construct.
        # 2. While possible:
        #    - Pick a random point in the set.
        #    - Remove it.
        #    - Check if the new set is maximal.
        #    - If yes, keep it (size reduced).
        #    - If no, try to add a random point from the "missing" points to restore maximality? 
        #      If we can restore it with 1 add, size stays same. Not helpful for minimization.
        #      If we can restore it with 0 adds (i.e., it was already maximal, which is false), size reduced.
        
        # Actually, the key is: A set is minimal maximal if no point can be removed while keeping it maximal.
        # So, we just need to find a maximal set where every point is "essential" for maximality.
        
        # Algorithm:
        # 1. Get a maximal set S.
        # 2. For each point p in S (random order):
        #    - Let S' = S \ {p}.
        #    - If S' is maximal, set S = S'. (We found a non-essential point).
        #    - Else, p is essential.
        # 3. Repeat until no change.
        
        # This is the "Reduce to Minimal" strategy. It's deterministic for a given S.
        # To escape local minima, we need to perturb S.
        
        # Perturbation: 
        # - Remove a random essential point p.
        # - Now S is not maximal.
        # - Add a random point q that can be added to S.
        # - Now S is valid again, but might not be maximal.
        # - Run "Maximize" step: While not maximal, add a random point that can be added.
        # - Run "Minimize" step: Remove non-essential points.
        
        current_s = list(current_set)
        
        # Minimize first
        changed = True
        while changed:
            changed = False
            random.shuffle(current_s)
            for p in current_s:
                if time.time() > deadline:
                    break
                trial_s = [x for x in current_s if x != p]
                if is_maximal(trial_s):
                    current_s = trial_s
                    changed = True
                    break # Restart loop to check new set
        
        if len(current_s) < best_size:
            best_size = len(current_s)
            best_set = list(current_s)
            
        # Try some perturbations if time permits
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Perturb: Remove a random point, then repair
            if len(current_s) > 0:
                p_to_remove = random.choice(current_s)
                trial_s = [x for x in current_s if x != p_to_remove]
                
                # Repair: Add points to make it maximal
                # We need to add points such that no more can be added.
                # This is a covering problem.
                # Greedy repair:
                added_any = True
                while added_any:
                    added_any = False
                    # Find all points that can be added
                    candidates = []
                    s_set = set(trial_s)
                    for q in range(NUM_POINTS):
                        if q not in s_set and can_add_point(trial_s, q):
                            candidates.append(q)
                    
                    if candidates:
                        # Pick a random candidate
                        q = random.choice(candidates)
                        trial_s.append(q)
                        added_any = True
                        # Optimization: We don't need to re-check all candidates from scratch immediately
                        # but for simplicity and correctness, we do.
                        # This loop is O(k * NUM_POINTS) per add.
                    
                # Now trial_s is maximal (or as close as we can get with greedy add).
                # Minimize it
                changed = True
                while changed:
                    changed = False
                    random.shuffle(trial_s)
                    for p in trial_s:
                        if time.time() > deadline:
                            break
                        t2 = [x for x in trial_s if x != p]
                        if is_maximal(t2):
                            trial_s = t2
                            changed = True
                            break
                
                # Accept if better or same (to allow plateau walking)
                if len(trial_s) <= len(current_s):
                    current_s = trial_s
                    if len(current_s) < best_size:
                        best_size = len(current_s)
                        best_set = list(current_s)

    if best_set is None:
        # Fallback to simple greedy if SA failed completely
        current_set = []
        for p_idx in range(NUM_POINTS):
            if can_add_point(current_set, p_idx):
                current_set.append(p_idx)
        return [POINTS[i] for i in current_set]
        
    return [POINTS[i] for i in best_set]

# EVOLVE-BLOCK-END
