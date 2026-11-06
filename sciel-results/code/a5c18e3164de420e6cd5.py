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

    # conflicts[x] = list of (u, v) pairs such that (u,v,x) is bad
    conflicts = [[] for _ in range(N)]
    for (a, b, c) in bad_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))

    # For fast validity: for each x, precompute bitmask of "partner" elements
    # blocked_mask[x] = bitmask where bit y is set if x and y together block some third element
    # Actually, let's precompute: for each pair (x, y), which z are blocked
    # blocked_by_pair[(x,y)] = bitmask of z such that (x,y,z) is a bad triple
    blocked_by_pair = {}
    for (a, b, c) in bad_triples:
        blocked_by_pair[(a, b)] = blocked_by_pair.get((a, b), 0) | (1 << c)
        blocked_by_pair[(b, a)] = blocked_by_pair.get((b, a), 0) | (1 << c)
        blocked_by_pair[(a, c)] = blocked_by_pair.get((a, c), 0) | (1 << b)
        blocked_by_pair[(c, a)] = blocked_by_pair.get((c, a), 0) | (1 << b)
        blocked_by_pair[(b, c)] = blocked_by_pair.get((b, c), 0) | (1 << a)
        blocked_by_pair[(c, b)] = blocked_by_pair.get((c, b), 0) | (1 << a)

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
        
        in_mask = [x for x in range(N) if (mask >> x) & 1]
        not_in_mask = [x for x in range(N) if not (mask >> x) & 1]
        
        random.shuffle(in_mask)
        
        for u in in_mask:
            if time.time() > deadline:
                return None
            temp_mask = mask & ~(1 << u)
            
            valid_candidates = []
            for x in not_in_mask:
                if is_valid_add(temp_mask, x):
                    valid_candidates.append(x)
            
            if len(valid_candidates) >= 2:
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
        
        # Use proper combinations for pairs
        pairs = list(combinations(in_mask, 2))
        random.shuffle(pairs)
        
        max_pairs = min(50, len(pairs))
        for (u, v) in pairs[:max_pairs]:
            if time.time() > deadline - 1.0:
                break
            
            temp_mask = mask & ~(1 << u) & ~(1 << v)
            
            valid_candidates = []
            for x in not_in_mask:
                if is_valid_add(temp_mask, x):
                    valid_candidates.append(x)
            
            if len(valid_candidates) >= 3:
                max_triples = 200
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
