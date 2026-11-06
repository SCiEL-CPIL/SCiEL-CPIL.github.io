import time
import random
import math
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    n = 128
    
    # First, find all bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == ab and _lcm(b, c) == ab:
                    bad_triples.append((a, b, c))
                    
    # Build conflict structures
    conflict_pairs = [set() for _ in range(n + 1)]
    for a, b, c in bad_triples:
        conflict_pairs[a].add((b, c))
        conflict_pairs[b].add((a, c))
        conflict_pairs[c].add((a, b))
        
    # Precompute conflict bitmasks for faster checking
    conflict_dict = [dict() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for u, v in conflict_pairs[x]:
            if u not in conflict_dict[x]:
                conflict_dict[x][u] = 0
            conflict_dict[x][u] |= (1 << (v - 1))
            if v not in conflict_dict[x]:
                conflict_dict[x][v] = 0
            conflict_dict[x][v] |= (1 << (u - 1))
    
    # Precompute flat conflict lists for faster can_add
    conflict_flat = [list() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for u, v_mask in conflict_dict[x].items():
            vm = v_mask
            while vm:
                lsb = vm & (-vm)
                v = lsb.bit_length()
                conflict_flat[x].append((u, v))
                vm &= vm - 1

    def can_add(x, mask):
        for u, v in conflict_flat[x]:
            if (mask & (1 << (u - 1))) and (mask & (1 << (v - 1))):
                return False
        return True

    def greedy(order, mask=0):
        selected = mask
        for x in order:
            if not (selected & (1 << (x - 1))):
                if can_add(x, selected):
                    selected |= (1 << (x - 1))
        return selected

    def popcount(x):
        return bin(x).count('1')
    
    def local_search(mask, max_iters=50):
        improved = True
        iters = 0
        
        while improved and time.time() < deadline - 2 and iters < max_iters:
            improved = False
            iters += 1
            
            # 1. Try to add any missing element
            for x in range(1, n + 1):
                if not (mask & (1 << (x - 1))):
                    if can_add(x, mask):
                        mask |= (1 << (x - 1))
                        improved = True
                        break
            if improved:
                continue

            # 2. Try 1-for-2 swap: remove x, add y, z
            best_new_mask = mask
            best_size = popcount(mask)
            s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]

            for x in s_list:
                if time.time() > deadline - 2:
                    break
                mask_no_x = mask & ~(1 << (x - 1))
                new_mask = mask_no_x
                added_count = 0
                for y in range(1, n + 1):
                    if y == x:
                        continue
                    if not (new_mask & (1 << (y - 1))):
                        if can_add(y, new_mask):
                            new_mask |= (1 << (y - 1))
                            added_count += 1
                            if added_count == 2:
                                break
                if added_count >= 2:
                    sz = popcount(new_mask)
                    if sz > best_size:
                        best_size = sz
                        best_new_mask = new_mask

            if best_size > popcount(mask):
                mask = best_new_mask
                improved = True
                continue

            # 3. Try 2-for-3 swap
            if time.time() < deadline - 3:
                s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                random.shuffle(s_list)
                best_new_mask = mask
                best_size = popcount(mask)
                for i in range(min(len(s_list), 20)):
                    if time.time() > deadline - 3:
                        break
                    x = s_list[i]
                    for j in range(i + 1, min(len(s_list), i + 5)):
                        y = s_list[j]
                        mask_no_xy = mask & ~(1 << (x - 1)) & ~(1 << (y - 1))
                        added_count = 0
                        new_mask = mask_no_xy
                        for z in range(1, n + 1):
                            if z == x or z == y:
                                continue
                            if not (new_mask & (1 << (z - 1))):
                                if can_add(z, new_mask):
                                    new_mask |= (1 << (z - 1))
                                    added_count += 1
                                    if added_count == 3:
                                        break
                        if added_count >= 3:
                            sz = popcount(new_mask)
                            if sz > best_size:
                                best_size = sz
                                best_new_mask = new_mask
                if best_size > popcount(mask):
                    mask = best_new_mask
                    improved = True
                    continue

            # 4. Random perturbation if stuck
            if not improved and iters > 5:
                if len(s_list) > 4:
                    remove_idx = random.sample(range(len(s_list)), 2)
                    perturb_mask = mask
                    for idx in remove_idx:
                        perturb_mask &= ~(1 << (s_list[idx] - 1))
                    for x in range(1, n + 1):
                        if not (perturb_mask & (1 << (x - 1))):
                            if can_add(x, perturb_mask):
                                perturb_mask |= (1 << (x - 1))
                    if popcount(perturb_mask) > popcount(mask):
                        mask = perturb_mask
                        improved = True
                        continue

            if not improved:
                break
                
        return mask

    def simulated_annealing(anchor_mask, variable_mask, iterations=5000):
        """
        Simulated annealing to optimize the set of elements to include, given fixed anchors.
        anchor_mask: elements that are fixed to be included
        variable_mask: elements that can be toggled
        Returns the best mask found.
        """
        if time.time() > deadline - 5:
            return anchor_mask | variable_mask  # Fallback: include all variables
        
        # Initialize: greedily add variables to anchors
        current_mask = anchor_mask
        for x in range(1, n + 1):
            if variable_mask & (1 << (x - 1)):
                if can_add(x, current_mask):
                    current_mask |= (1 << (x - 1))
        
        best_mask = current_mask
        best_size = popcount(current_mask)
        current_size = best_size
        
        # Simulated annealing parameters
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        
        for it in range(iterations):
            if time.time() > deadline - 5:
                break
            
            # Random move: toggle a variable
            # Pick a random element from variable_mask
            temp = variable_mask
            if temp == 0:
                break
            # Pick random bit position
            r = random.randint(1, n)
            if temp & (1 << (r - 1)):
                x = r
            else:
                # Find a set bit
                temp2 = temp
                while temp2:
                    lsb = temp2 & (-temp2)
                    x = lsb.bit_length()
                    break
                temp2 = temp2 - lsb
            
            if current_mask & (1 << (x - 1)):
                # Try removing x
                new_mask = current_mask & ~(1 << (x - 1))
                new_size = current_size - 1
                delta = -1
            else:
                # Try adding x
                if can_add(x, current_mask):
                    new_mask = current_mask | (1 << (x - 1))
                    new_size = current_size + 1
                    delta = 1
                else:
                    # Can't add x due to conflicts, skip this move
                    T *= cooling
                    continue
            
            # Accept if better, or with probability exp(-delta/T)
            if delta > 0:
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
            else:
                # delta <= 0, accept with probability
                if random.random() < math.exp(-delta / T):
                    current_mask = new_mask
                    current_size = new_size
            
            T *= cooling
        
        return best_mask

    def find_best_with_anchors(anchor_count, num_trials=50):
        """
        Try multiple anchor sets of given size, run simulated annealing for each.
        Returns best mask and size.
        """
        best_mask = 0
        best_size = 0
        
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
            
            # Choose random anchors
            # Prefer low-degree elements for anchors (more flexible)
            all_elements = list(range(1, n + 1))
            random.shuffle(all_elements)
            
            # Sort by degree (ascending) and pick from lower degrees
            degrees = [len(conflict_pairs[k]) for k in all_elements]
            indexed = list(zip(degrees, all_elements))
            indexed.sort()
            candidates = [x for d, x in indexed]
            
            # Pick anchor_count random candidates, biased toward low degree
            anchors = random.sample(candidates[:min(len(candidates), anchor_count * 3)], 
                                   min(anchor_count, len(candidates)))
            
            anchor_mask = 0
            for a in anchors:
                anchor_mask |= (1 << (a - 1))
            
            # Variable mask: all elements not in anchors
            variable_mask = ((1 << n) - 1) & ~anchor_mask
            
            # Run simulated annealing
            sa_mask = simulated_annealing(anchor_mask, variable_mask, iterations=3000)
            
            # Polish with local search
            sa_mask = local_search(sa_mask, max_iters=20)
            
            size = popcount(sa_mask)
            if size > best_size:
                best_size = size
                best_mask = sa_mask
        
        return best_mask, best_size

    best_mask = 0
    best_size = 0
    
    # Precompute degrees for sorting
    degrees = [len(conflict_pairs[k]) for k in range(n + 1)]

    # Run multiple randomized greedy passes first to get a good initial solution
    num_restarts = 200

    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break

        strategy = i % 6
        if strategy == 0:
            order = sorted(range(1, n + 1), key=lambda x: degrees[x] + random.random() * 0.1)
        elif strategy == 1:
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x] + random.random() * 0.1)
        elif strategy == 2:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif strategy == 3:
            order = list(range(n, 0, -1))
            random.shuffle(order)
        elif strategy == 4:
            order = sorted(range(1, n + 1), key=lambda x: degrees[x] / max(1, x) + random.random() * 0.01)
        else:
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x] / max(1, x) + random.random() * 0.01)

        mask = greedy(order)
        mask = local_search(mask)

        size = popcount(mask)
        if size > best_size:
            best_size = size
            best_mask = mask
            
    # Now try anchor-based simulated annealing with different anchor counts
    if time.time() < deadline - 15:
        for anchor_count in [5, 10, 15, 20]:
            if time.time() > deadline - 15:
                break
            sa_mask, sa_size = find_best_with_anchors(anchor_count, num_trials=30)
            if sa_size > best_size:
                best_size = sa_size
                best_mask = sa_mask
    
    # Final polish
    if time.time() < deadline - 2:
        best_mask = local_search(best_mask, max_iters=100)
    
    result = [i for i in range(1, n + 1) if (best_mask & (1 << (i - 1)))]
    
    if not result:
        result = [1]
        
    return result
