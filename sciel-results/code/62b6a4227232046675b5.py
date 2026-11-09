"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    """
    Construct a subset of [1,256] with no 4-term AP, maximizing 3-term APs.
    Uses Simulated Annealing with bitset representation for speed.
    """
    deadline = time.time() + 85  # safety margin before 90s kill
    N = 256
    # Bitmask: bit i represents element i+1 (1-indexed)
    # So element x is represented by bit (x-1)
    
    # Precompute 4-AP conflict masks: for each element x, mask of elements y
    # such that {x,y} participates in a 4-AP with two other elements already present
    # Actually, we need: for a set S and element x not in S, check if S∪{x} has a 4-AP
    # A 4-AP is a, a+d, a+2d, a+3d. x must be one of these four positions.
    # For each x, precompute which triples (positions) would form a 4-AP.
    
    # For efficiency, precompute for each x the list of (d, pos) pairs where
    # x is at position pos in a 4-AP with difference d, and the other 3 elements.
    
    # Let's use a different approach: for each x, precompute a bitmask conflict[x]
    # such that if we have set S (as bitmask), then adding x creates a 4-AP iff
    # there exists a 4-AP containing x where the other 3 elements are all in S.
    
    # For element x (1-indexed), it can be in a 4-AP at position 0,1,2,3:
    # pos 0: x, x+d, x+2d, x+3d -> need x+d, x+2d, x+3d in S
    # pos 1: x-d, x, x+d, x+2d -> need x-d, x+d, x+2d in S  
    # pos 2: x-2d, x-d, x, x+d -> need x-2d, x-d, x+d in S
    # pos 3: x-3d, x-2d, x-d, x -> need x-3d, x-2d, x-d in S
    
    # For each x, build a list of bitmasks representing the "required other elements"
    # for each potential 4-AP that x could be part of.
    
    # Precompute: conflicts[x] = list of bitmasks (each mask has 3 bits set)
    # Adding x to S creates 4-AP iff any mask in conflicts[x] is a subset of S.
    
    conflicts = [[] for _ in range(N+1)]
    for x in range(1, N+1):
        # Position 0: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            e1, e2, e3 = x+d, x+2*d, x+3*d
            if e3 <= N:
                mask = (1 << (e1-1)) | (1 << (e2-1)) | (1 << (e3-1))
                conflicts[x].append(mask)
        # Position 1: x-d, x, x+d, x+2d
        for d in range(1, min(x-1, N-1-x) + 1):
            e1, e2, e3 = x-d, x+d, x+2*d
            if e1 >= 1 and e3 <= N:
                mask = (1 << (e1-1)) | (1 << (e2-1)) | (1 << (e3-1))
                conflicts[x].append(mask)
        # Position 2: x-2d, x-d, x, x+d
        for d in range(1, min((x-1)//2, N-1-x) + 1):
            e1, e2, e3 = x-2*d, x-d, x+d
            if e1 >= 1 and e3 <= N:
                mask = (1 << (e1-1)) | (1 << (e2-1)) | (1 << (e3-1))
                conflicts[x].append(mask)
        # Position 3: x-3d, x-2d, x-d, x
        for d in range(1, (x-1)//3 + 1):
            e1, e2, e3 = x-3*d, x-2*d, x-d
            if e1 >= 1:
                mask = (1 << (e1-1)) | (1 << (e2-1)) | (1 << (e3-1))
                conflicts[x].append(mask)
    
    def has_4ap_with_mask(S_mask, x):
        """Check if adding element x (1-indexed) to set S (bitmask) creates a 4-AP."""
        for mask in conflicts[x]:
            if (S_mask & mask) == mask:
                return True
        return False
    
    # Precompute 3-AP contributions: for each x, the bitmask of pairs that would
    # form a 3-AP with x. We need to count 3-APs efficiently.
    # A 3-AP is a, a+d, a+2d. For element x, it can be:
    # pos 0: x, x+d, x+2d -> need x+d and x+2d
    # pos 1: x-d, x, x+d -> need x-d and x+d
    # pos 2: x-2d, x-d, x -> need x-2d and x-d
    
    # For counting 3-APs in a set, we can precompute for each x the list of pairs.
    # But for incremental updates, we need gain/loss when adding/removing x.
    
    # Precompute: for each x, list of (y, z) pairs such that {x,y,z} forms a 3-AP
    # and x is the "distinguished" element. Actually, let's precompute:
    # three_ap_pairs[x] = list of (y, z) where {x,y,z} is a 3-AP with x being
    # one of the three, and y < z (to avoid double counting issues)
    
    # Actually, for counting total 3-APs, each 3-AP {a, a+d, a+2d} is counted once.
    # When we add element x, the new 3-APs are those where x is one of the three terms.
    
    # Precompute for each x:
    # - x as first: pairs (x+d, x+2d) for valid d
    # - x as middle: pairs (x-d, x+d) for valid d  
    # - x as last: pairs (x-2d, x-d) for valid d
    
    # For incremental count: gain(x, S) = number of pairs (y,z) in three_ap_pairs[x]
    # such that y,z both in S.
    
    # Let's precompute three_ap_neighbors[x] = list of (y, z) where y < z and
    # {x,y,z} forms a 3-AP.
    
    three_ap_neighbors = [[] for _ in range(N+1)]
    for x in range(1, N+1):
        seen = set()
        # x as first term: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            y, z = x+d, x+2*d
            if z <= N:
                pair = (min(y,z), max(y,z))
                if pair not in seen:
                    seen.add(pair)
                    three_ap_neighbors[x].append(pair)
        # x as middle term: x-d, x, x+d
        for d in range(1, min(x-1, N-x) + 1):
            y, z = x-d, x+d
            if y >= 1:
                pair = (min(y,z), max(y,z))
                if pair not in seen:
                    seen.add(pair)
                    three_ap_neighbors[x].append(pair)
        # x as last term: x-2d, x-d, x
        for d in range(1, (x-1)//2 + 1):
            y, z = x-2*d, x-d
            if y >= 1:
                pair = (min(y,z), max(y,z))
                if pair not in seen:
                    seen.add(pair)
                    three_ap_neighbors[x].append(pair)
    
    def count_3aps_mask(S_mask):
        """Count 3-APs in set represented by bitmask."""
        count = 0
        for x in range(1, N+1):
            if not (S_mask & (1 << (x-1))):
                continue
            for y, z in three_ap_neighbors[x]:
                if (S_mask & (1 << (y-1))) and (S_mask & (1 << (z-1))):
                    count += 1
        # Each 3-AP is counted 3 times (once for each element as "x")
        # Wait, no. Let me reconsider.
        # For a 3-AP {a, a+d, a+2d}:
        # When x=a: pair is (a+d, a+2d) -> counted if both in S
        # When x=a+d: pair is (a, a+2d) -> counted if both in S
        # When x=a+2d: pair is (a, a+d) -> counted if both in S
        # So each 3-AP is counted 3 times.
        return count // 3
    
    def gain_3aps(S_mask, x):
        """Count new 3-APs created by adding x to S."""
        gain = 0
        for y, z in three_ap_neighbors[x]:
            if (S_mask & (1 << (y-1))) and (S_mask & (1 << (z-1))):
                gain += 1
        return gain
    
    def loss_3aps(S_mask, x):
        """Count 3-APs lost by removing x from S."""
        loss = 0
        for y, z in three_ap_neighbors[x]:
            if (S_mask & (1 << (y-1))) and (S_mask & (1 << (z-1))):
                loss += 1
        return loss
    
    # Greedy construction to get initial solution
    def greedy_construct(seed=42):
        rng = random.Random(seed)
        S_mask = 0
        order = list(range(1, N+1))
        rng.shuffle(order)
        
        for x in order:
            if not has_4ap_with_mask(S_mask, x):
                S_mask |= (1 << (x-1))
        
        return S_mask
    
    # Simulated Annealing
    def simulated_annealing(initial_mask, time_limit):
        current_mask = initial_mask
        current_count = count_3aps_mask(current_mask)
        best_mask = current_mask
        best_count = current_count
        
        T = 5.0  # initial temperature
        cooling = 0.999
        min_T = 0.01
        rng = random.Random(12345)
        
        start_time = time.time()
        no_improve_count = 0
        
        while T > min_T:
            if time.time() > time_limit:
                break
            
            # Random move: swap or add or remove
            move_type = rng.random()
            
            if move_type < 0.3:
                # Try adding an element
                candidates = [x for x in range(1, N+1) 
                             if not (current_mask & (1 << (x-1)))
                             and not has_4ap_with_mask(current_mask, x)]
                if not candidates:
                    T *= cooling
                    continue
                x = rng.choice(candidates)
                new_mask = current_mask | (1 << (x-1))
                delta = gain_3aps(current_mask, x)
                new_count = current_count + delta
            elif move_type < 0.6:
                # Try removing an element
                in_set = [x for x in range(1, N+1) if (current_mask & (1 << (x-1)))]
                if not in_set:
                    T *= cooling
                    continue
                x = rng.choice(in_set)
                new_mask = current_mask & ~(1 << (x-1))
                delta = -loss_3aps(current_mask, x)
                new_count = current_count + delta
            else:
                # Try swapping: remove x, add y
                in_set = [x for x in range(1, N+1) if (current_mask & (1 << (x-1)))]
                not_in_set = [x for x in range(1, N+1) if not (current_mask & (1 << (x-1)))]
                if not in_set or not not_in_set:
                    T *= cooling
                    continue
                x = rng.choice(in_set)
                # Try y that doesn't create 4-AP after removing x
                temp_mask = current_mask & ~(1 << (x-1))
                valid_ys = [y for y in not_in_set if not has_4ap_with_mask(temp_mask, y)]
                if not valid_ys:
                    T *= cooling
                    continue
                y = rng.choice(valid_ys)
                new_mask = temp_mask | (1 << (y-1))
                # Compute delta
                delta = -loss_3aps(current_mask, x) + gain_3aps(temp_mask, y)
                new_count = current_count + delta
            
            # Accept or reject
            if new_count > current_count:
                current_mask = new_mask
                current_count = new_count
                no_improve_count = 0
                if current_count > best_count:
                    best_count = current_count
                    best_mask = current_mask
            else:
                # Metropolis acceptance
                prob = math.exp(delta / T) if delta < 0 else 1.0
                if rng.random() < prob:
                    current_mask = new_mask
                    current_count = new_count
                    no_improve_count += 1
                else:
                    no_improve_count += 1
            
            T *= cooling
        
        return best_mask, best_count
    
    # Get initial solution
    best_mask = greedy_construct(42)
    best_count = count_3aps_mask(best_mask)
    
    # Run simulated annealing with multiple restarts
    start_time = time.time()
    remaining = deadline - start_time
    
    # First SA run with most time
    sa_time1 = remaining * 0.5
    best_mask, best_count = simulated_annealing(best_mask, start_time + sa_time1)
    
    # Second SA run from best
    start_time2 = time.time()
    remaining2 = deadline - start_time2
    if remaining2 > 5:
        sa_time2 = remaining2 * 0.6
        m, c = simulated_annealing(best_mask, start_time2 + sa_time2)
        if c > best_count:
            best_mask, best_count = m, c
    
    # Third SA run
    start_time3 = time.time()
    remaining3 = deadline - start_time3
    if remaining3 > 5:
        sa_time3 = remaining3 * 0.7
        m, c = simulated_annealing(best_mask, start_time3 + sa_time3)
        if c > best_count:
            best_mask, best_count = m, c
    
    # Convert bitmask to sorted list
    result = [x for x in range(1, N+1) if (best_mask & (1 << (x-1)))]
    
    # Safety check: verify no 4-AP
    def verify_no_4ap(A):
        s = set(A)
        for a in A:
            for d in range(1, (N - a) // 3 + 1):
                if a+d in s and a+2*d in s and a+3*d in s:
                    return False
        return True
    
    if not verify_no_4ap(result):
        # Fallback to greedy
        result = [x for x in range(1, N+1) if (greedy_construct(42) & (1 << (x-1)))]
    
    return sorted(result)
# EVOLVE-BLOCK-END
