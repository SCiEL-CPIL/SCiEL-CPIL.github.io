import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Extended deadline for more search time
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) configurations
    # A K(2,2,2) is formed by 3 disjoint pairs.
    # There are 15 ways to partition 6 items into 3 pairs.
    # Total configs: C(16,6) * 15
    k222_list = []
    triple_to_k222s = [[] for _ in range(num_triples)]
    config_triples = [] # List of lists of triple indices
    
    for points in combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = points
        # Generate 15 pairings
        pairings = [
            ((p0,p1),(p2,p3),(p4,p5)),
            ((p0,p1),(p2,p4),(p3,p5)),
            ((p0,p1),(p2,p5),(p3,p4)),
            ((p0,p2),(p1,p3),(p4,p5)),
            ((p0,p2),(p1,p4),(p3,p5)),
            ((p0,p2),(p1,p5),(p3,p4)),
            ((p0,p3),(p1,p2),(p4,p5)),
            ((p0,p3),(p1,p4),(p2,p5)),
            ((p0,p3),(p1,p5),(p2,p4)),
            ((p0,p4),(p1,p2),(p3,p5)),
            ((p0,p4),(p1,p3),(p2,p5)),
            ((p0,p4),(p1,p5),(p2,p3)),
            ((p0,p5),(p1,p2),(p3,p4)),
            ((p0,p5),(p1,p3),(p2,p4)),
            ((p0,p5),(p1,p4),(p2,p3)),
        ]
        
        for pair1, pair2, pair3 in pairings:
            transversals = []
            for x in pair1:
                for y in pair2:
                    for z in pair3:
                        transversals.append((x, y, z))
            
            trans_sorted = [tuple(sorted(t)) for t in transversals]
            trans_indices = [triple_to_idx[t] for t in trans_sorted]
            
            k_idx = len(k222_list)
            k222_list.append(set(trans_indices))
            config_triples.append(trans_indices)
            
            for idx in trans_indices:
                triple_to_k222s[idx].append(k_idx)
    
    num_k222s = len(k222_list)
    
    def time_left():
        return deadline - time.time()

    def greedy_init():
        active = [False] * num_triples
        counts = [0] * num_k222s
        current_size = 0
        order = list(range(num_triples))
        random.shuffle(order)
        
        for t_idx in order:
            valid = True
            for k_idx in triple_to_k222s[t_idx]:
                if counts[k_idx] == 7:
                    valid = False
                    break
            if valid:
                active[t_idx] = True
                current_size += 1
                for k_idx in triple_to_k222s[t_idx]:
                    counts[k_idx] += 1
        return active, counts, current_size

    def is_addable(t_idx, counts):
        for k_idx in triple_to_k222s[t_idx]:
            if counts[k_idx] == 7:
                return False
        return True

    def get_blockers(t_idx, counts):
        """Returns list of config indices c where counts[c] == 7"""
        blockers = []
        for k_idx in triple_to_k222s[t_idx]:
            if counts[k_idx] == 7:
                blockers.append(k_idx)
        return blockers

    def local_search(active, counts, current_size):
        improved = True
        while improved and time_left() > 2:
            improved = False
            
            # 1. Try to add any edge
            addable = []
            for i in range(num_triples):
                if not active[i] and is_addable(i, counts):
                    addable.append(i)
                    
            if addable:
                # Pick a random addable edge
                t_idx = random.choice(addable)
                active[t_idx] = True
                current_size += 1
                for k_idx in triple_to_k222s[t_idx]:
                    counts[k_idx] += 1
                improved = True
                continue # Restart search to find more adds
            
            # 2. If no addable, try swaps
            inactive_edges = [i for i in range(num_triples) if not active[i]]
            if not inactive_edges:
                break # Full set (unlikely)
                
            # Try a few random swaps
            found_swap = False
            # Increase attempts to escape local minima more effectively
            num_swap_attempts = 100
            for _ in range(num_swap_attempts):
                if time_left() < 1: break
                t_idx = random.choice(inactive_edges)
                
                # Check if t_idx is blocked
                blockers = get_blockers(t_idx, counts)
                if not blockers:
                    # Should have been added in step 1, but just in case
                    active[t_idx] = True
                    current_size += 1
                    for k_idx in triple_to_k222s[t_idx]:
                        counts[k_idx] += 1
                    improved = True
                    found_swap = True
                    break
                
                # Strategy: Pick a random blocker, pick a random active triple in that blocker to remove
                # This is more exploratory than intersection-based swaps
                random.shuffle(blockers)
                swap_success = False
                for c_idx in blockers:
                    # Get active triples in this config
                    # config_triples[c_idx] contains 8 triple indices
                    # We need one that is active
                    # Optimization: Precompute active triples in config? No, just check.
                    # Since config size is small (8), linear scan is fine.
                    candidates = []
                    for t_rem in config_triples[c_idx]:
                        if active[t_rem]:
                            candidates.append(t_rem)
                    
                    if not candidates:
                        continue
                        
                    # Try removing one of these candidates
                    random.shuffle(candidates)
                    for t_rem in candidates:
                        # Temporarily remove t_rem
                        active[t_rem] = False
                        for k_idx in triple_to_k222s[t_rem]:
                            counts[k_idx] -= 1
                        
                        # Check if t_idx is now addable
                        if is_addable(t_idx, counts):
                            # Perform swap
                            active[t_idx] = True
                            for k_idx in triple_to_k222s[t_idx]:
                                counts[k_idx] += 1
                            # Note: current_size remains same
                            improved = True
                            found_swap = True
                            swap_success = True
                            break
                        else:
                            # Restore t_rem
                            active[t_rem] = True
                            for k_idx in triple_to_k222s[t_rem]:
                                counts[k_idx] += 1
                        
                        if swap_success:
                            break
                    
                    if swap_success:
                        break
                
                if found_swap:
                    break # Restart local search loop to find adds
                    
            if not found_swap:
                # No swap found in this batch of attempts
                # Break local search phase to allow ILS perturbation
                break

        return active, counts, current_size

    best_size = 0
    best_active = None
    best_counts = None
    
    # ILS Loop
    # 1. Greedy Init
    active, counts, current_size = greedy_init()
    active, counts, current_size = local_search(active, counts, current_size)
    
    if current_size > best_size:
        best_size = current_size
        best_active = active[:]
        best_counts = counts[:]
        
    # 2. Iterated Local Search
    while time_left() > 5:
        # Perturb
        # Remove k random edges
        # Vary k slightly to explore different basin sizes
        k = random.randint(5, 20)
        active_edges = [i for i in range(num_triples) if active[i]]
        if len(active_edges) < k:
            k = len(active_edges)
            
        # Shuffle active edges and remove first k
        random.shuffle(active_edges)
        for i in range(k):
            t_idx = active_edges[i]
            active[t_idx] = False
            current_size -= 1
            for k_idx in triple_to_k222s[t_idx]:
                counts[k_idx] -= 1
                
        # Local Search on perturbed solution
        active, counts, current_size = local_search(active, counts, current_size)
        
        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            best_counts = counts[:]
            
    # Construct result
    if best_active is None:
        active, counts, current_size = greedy_init()
        best_active = active
        
    result = []
    for i in range(num_triples):
        if best_active[i]:
            result.append(list(all_triples[i]))
            
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Number of triples: {len(result)}")
