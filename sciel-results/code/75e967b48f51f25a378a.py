import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 128
    # Use 0-indexed internally: element x -> index x-1
    # Bitmask: bit i set means element (i+1) is in the set

    # Precompute bad triples and conflicts
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                g_ab = gcd(a, b)
                g_bc = gcd(b, c)
                g_ac = gcd(a, c)
                if g_ab == g_bc == g_ac:
                    bad_triples.append((a - 1, b - 1, c - 1))  # 0-indexed

    # conflicts[x] = bitmask of elements that conflict with x when paired with another in set
    # Actually, let's use: for each x, store list of (u, v) pairs such that (u,v,x) is bad
    # But for bitmask speed, let's precompute for each x a list of (u,v) pairs
    conflicts = [[] for _ in range(N)]
    for (a, b, c) in bad_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))

    # For faster checking: for each x, precompute a list of (u, v) where u < v
    # and also build a bitmask-based check

    def is_valid_add(mask, x):
        """Check if adding x (0-indexed) to set represented by mask is valid."""
        for (u, v) in conflicts[x]:
            if (mask >> u) & 1 and (mask >> v) & 1:
                return False
        return True

    # Greedy construction with multiple orders
    random.seed(42)
    best_mask = 0
    best_size = 0

    def greedy(order):
        mask = 0
        size = 0
        for x in order:
            if is_valid_add(mask, x):
                mask |= (1 << x)
                size += 1
        return mask, size

    # Compute degree
    degree = [len(conflicts[x]) for x in range(N)]

    # Create various orderings
    all_elements = list(range(N))
    orders = [
        all_elements[:],
        all_elements[::-1],
        sorted(all_elements, key=lambda x: degree[x]),
        sorted(all_elements, key=lambda x: -degree[x]),
        sorted(all_elements, key=lambda x: (degree[x], -x)),
        sorted(all_elements, key=lambda x: (-degree[x], x)),
    ]

    # Random orders
    for _ in range(20):
        random_order = all_elements[:]
        random.shuffle(random_order)
        orders.append(random_order)

    # Prime-heavy
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [x for x in range(N) if is_prime(x + 1)]
    composites = [x for x in range(N) if not is_prime(x + 1)]
    orders.append(primes + composites)
    orders.append(composites + primes)

    for order in orders:
        mask, size = greedy(order)
        if size > best_size:
            best_size = size
            best_mask = mask

    # Local search using bitmask operations
    def try_add_one(mask):
        """Try to add one element. Return new mask if successful, else None."""
        for x in range(N):
            if not (mask >> x) & 1:
                if is_valid_add(mask, x):
                    return mask | (1 << x)
        return None

    def try_swap_1_for_2(mask):
        """Try to remove 1 element and add 2."""
        if time.time() > deadline:
            return None
        
        # Get list of elements in mask
        in_mask = [x for x in range(N) if (mask >> x) & 1]
        not_in_mask = [x for x in range(N) if not (mask >> x) & 1]
        
        random.shuffle(in_mask)
        
        for u in in_mask:
            if time.time() > deadline:
                return None
            temp_mask = mask & ~(1 << u)
            
            # Find candidates that can be added
            valid_candidates = []
            for x in not_in_mask:
                if is_valid_add(temp_mask, x):
                    valid_candidates.append(x)
            
            if len(valid_candidates) >= 2:
                # Try all pairs
                for i in range(len(valid_candidates)):
                    x = valid_candidates[i]
                    temp_with_x = temp_mask | (1 << x)
                    for j in range(i + 1, len(valid_candidates)):
                        y = valid_candidates[j]
                        if is_valid_add(temp_with_x, y):
                            return temp_mask | (1 << x) | (1 << y)
        return None

    def try_swap_2_for_3(mask):
        """Try to remove 2 elements and add 3."""
        if time.time() > deadline - 2.0:
            return None
        
        in_mask = [x for x in range(N) if (mask >> x) & 1]
        not_in_mask = [x for x in range(N) if not (mask >> x) & 1]
        
        random.shuffle(in_mask)
        
        max_pairs = 30
        for pair_idx in range(min(max_pairs, len(in_mask) * (len(in_mask) - 1) // 2)):
            if time.time() > deadline - 1.0:
                break
            
            # Get pair_idx-th combination
            i = pair_idx // (len(in_mask) - 1)
            j = pair_idx % (len(in_mask) - 1)
            if i >= len(in_mask) or j >= len(in_mask) or i == j:
                continue
            u, v = in_mask[i], in_mask[j]
            if u == v:
                continue
            
            temp_mask = mask & ~(1 << u) & ~(1 << v)
            
            # Find candidates
            valid_candidates = []
            for x in not_in_mask:
                if is_valid_add(temp_mask, x):
                    valid_candidates.append(x)
            
            if len(valid_candidates) >= 3:
                # Try limited triples
                max_triples = 100
                triple_count = 0
                for i in range(len(valid_candidates)):
                    if triple_count >= max_triples:
                        break
                    x = valid_candidates[i]
                    temp_x = temp_mask | (1 << x)
                    for j in range(i + 1, len(valid_candidates)):
                        if triple_count >= max_triples:
                            break
                        y = valid_candidates[j]
                        temp_xy = temp_x | (1 << y)
                        for k in range(j + 1, len(valid_candidates)):
                            if triple_count >= max_triples:
                                break
                            z = valid_candidates[k]
                            if is_valid_add(temp_xy, z):
                                return temp_mask | (1 << x) | (1 << y) | (1 << z)
                            triple_count += 1
        return None

    # Main local search loop
    S_mask = best_mask
    current_size = best_size
    stuck_count = 0
    no_improve_count = 0

    while time.time() < deadline:
        improved = False

        # N0: Add 1
        new_mask = try_add_one(S_mask)
        if new_mask is not None:
            S_mask = new_mask
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_mask = S_mask
                no_improve_count = 0
            else:
                no_improve_count += 1
            stuck_count = 0
            improved = True
            continue

        # N1: 1-for-2
        new_mask = try_swap_1_for_2(S_mask)
        if new_mask is not None:
            S_mask = new_mask
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_mask = S_mask
                no_improve_count = 0
            else:
                no_improve_count += 1
            stuck_count = 0
            improved = True
            continue

        # N2: 2-for-3
        new_mask = try_swap_2_for_3(S_mask)
        if new_mask is not None:
            S_mask = new_mask
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_mask = S_mask
                no_improve_count = 0
            else:
                no_improve_count += 1
            stuck_count = 0
            improved = True
            continue

        # 1-for-1 swap to change structure
        improved_structure = False
        in_mask = [x for x in range(N) if (S_mask >> x) & 1]
        not_in_mask = [x for x in range(N) if not (S_mask >> x) & 1]
        random.shuffle(in_mask)

        for u in in_mask:
            if time.time() > deadline:
                break
            temp_mask = S_mask & ~(1 << u)
            for v in not_in_mask:
                if is_valid_add(temp_mask, v):
                    S_mask = temp_mask | (1 << v)
                    improved_structure = True
                    break
            if improved_structure:
                break

        if improved_structure:
            stuck_count = 0
            no_improve_count += 1
        else:
            stuck_count += 1
            no_improve_count += 1
            if stuck_count > 15 or no_improve_count > 50:
                # Random restart
                S_mask = best_mask
                stuck_count = 0
                no_improve_count = 0

                # Perturb with 1-3 random swaps
                in_mask = [x for x in range(N) if (S_mask >> x) & 1]
                not_in_mask = [x for x in range(N) if not (S_mask >> x) & 1]
                num_swaps = random.randint(1, 3)
                for _ in range(num_swaps):
                    if not in_mask or not not_in_mask:
                        break
                    u = random.choice(in_mask)
                    v = random.choice(not_in_mask)
                    temp_mask = S_mask & ~(1 << u)
                    if is_valid_add(temp_mask, v):
                        S_mask = temp_mask | (1 << v)
                        in_mask = [x for x in range(N) if (S_mask >> x) & 1]
                        not_in_mask = [x for x in range(N) if not (S_mask >> x) & 1]

    # Convert best_mask back to list of 1-indexed elements
    result = [x + 1 for x in range(N) if (best_mask >> x) & 1]
    return result if result else [1]

    # Optimized local search functions
    def try_add_one(S):
        """Try to add one element. Return new S if successful, else None."""
        for x in range(1, N + 1):
            if x not in S:
                if is_valid_add(S, x):
                    new_S = S.copy()
                    new_S.add(x)
                    return new_S
        return None

    def try_swap_1_for_2(S):
        """Try to remove 1 element and add 2. Optimized version."""
        S_list = list(S)
        random.shuffle(S_list)
        candidates = [x for x in range(1, N + 1) if x not in S]
        
        for u in S_list:
            if time.time() > deadline:
                return None
            temp_S = S - {u}
            # Find all candidates that can be added to temp_S
            valid_candidates = []
            for x in candidates:
                if is_valid_add(temp_S, x):
                    valid_candidates.append(x)
            
            # Try all pairs from valid_candidates
            if len(valid_candidates) >= 2:
                for i in range(len(valid_candidates)):
                    x = valid_candidates[i]
                    temp_S_with_x = temp_S | {x}
                    for j in range(i + 1, len(valid_candidates)):
                        y = valid_candidates[j]
                        if is_valid_add(temp_S_with_x, y):
                            new_S = temp_S | {x, y}
                            return new_S
        return None

    def try_swap_2_for_3(S):
        """Try to remove 2 elements and add 3. Limited version for time efficiency."""
        if time.time() > deadline - 2.0:
            return None
        
        S_list = list(S)
        random.shuffle(S_list)
        candidates = [x for x in range(1, N + 1) if x not in S]
        
        # Limit the search to avoid timeout
        max_pairs_to_check = 20
        
        for pair_idx, (u, v) in enumerate(combinations(S_list, 2)):
            if pair_idx >= max_pairs_to_check:
                break
            if time.time() > deadline - 1.0:
                break
            
            temp_S = S - {u, v}
            # Find candidates that can be added to temp_S
            valid_candidates = []
            for x in candidates:
                if is_valid_add(temp_S, x):
                    valid_candidates.append(x)
            
            # Try all triples from valid_candidates (limited)
            if len(valid_candidates) >= 3:
                # Limit triple search
                max_triples = 50
                triple_count = 0
                for i in range(len(valid_candidates)):
                    if triple_count >= max_triples:
                        break
                    x = valid_candidates[i]
                    temp_S_with_x = temp_S | {x}
                    for j in range(i + 1, len(valid_candidates)):
                        if triple_count >= max_triples:
                            break
                        y = valid_candidates[j]
                        temp_S_with_xy = temp_S_with_x | {y}
                        for k in range(j + 1, len(valid_candidates)):
                            if triple_count >= max_triples:
                                break
                            z = valid_candidates[k]
                            if is_valid_add(temp_S_with_xy, z):
                                new_S = temp_S | {x, y, z}
                                return new_S
                            triple_count += 1
        return None

    # VNS Loop with multiple neighborhoods
    stuck_count = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # N0: Add 1
        new_S = try_add_one(S)
        if new_S:
            S = new_S
            current_size = len(S)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            stuck_count = 0
            improved = True
            continue
        
        # N1: 1-for-2
        new_S = try_swap_1_for_2(S)
        if new_S:
            S = new_S
            current_size = len(S)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            stuck_count = 0
            improved = True
            continue
        
        # N2: 2-for-3
        new_S = try_swap_2_for_3(S)
        if new_S:
            S = new_S
            current_size = len(S)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            stuck_count = 0
            improved = True
            continue
        
        # If stuck, try a 1-for-1 swap to change the structure (size remains same)
        improved_structure = False
        S_list = list(S)
        random.shuffle(S_list)
        candidates = [x for x in range(1, N + 1) if x not in S]
        
        for u in S_list:
            if time.time() > deadline:
                break
            temp_S = S - {u}
            for v in candidates:
                if is_valid_add(temp_S, v):
                    # This swap is valid
                    S = temp_S | {v}
                    improved_structure = True
                    break
            if improved_structure:
                break
        
        if improved_structure:
            stuck_count = 0
            no_improve_count += 1
        else:
            stuck_count += 1
            no_improve_count += 1
            if stuck_count > 15 or no_improve_count > 50:
                # Random restart from best with perturbation
                S = best_set.copy()
                stuck_count = 0
                no_improve_count = 0
                
                # Apply 1-3 random 1-for-1 swaps to perturb
                S_list = list(S)
                num_swaps = random.randint(1, 3)
                for _ in range(num_swaps):
                    if not S_list:
                        break
                    u = random.choice(S_list)
                    v = random.randint(1, N)
                    if v not in S:
                        temp_S = S - {u}
                        if is_valid_add(temp_S, v):
                            S = temp_S | {v}
                            S_list = list(S)

    return list(best_set)
