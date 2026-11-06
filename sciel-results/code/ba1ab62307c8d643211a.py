import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit
    
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
    config_triples = []  # List of lists of triple indices
    
    for points in combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = points
        # Generate 15 pairings
        pairings = [
            ((p0, p1), (p2, p3), (p4, p5)),
            ((p0, p1), (p2, p4), (p3, p5)),
            ((p0, p1), (p2, p5), (p3, p4)),
            ((p0, p2), (p1, p3), (p4, p5)),
            ((p0, p2), (p1, p4), (p3, p5)),
            ((p0, p2), (p1, p5), (p3, p4)),
            ((p0, p3), (p1, p2), (p4, p5)),
            ((p0, p3), (p1, p4), (p2, p5)),
            ((p0, p3), (p1, p5), (p2, p4)),
            ((p0, p4), (p1, p2), (p3, p5)),
            ((p0, p4), (p1, p3), (p2, p5)),
            ((p0, p4), (p1, p5), (p2, p3)),
            ((p0, p5), (p1, p2), (p3, p4)),
            ((p0, p5), (p1, p3), (p2, p4)),
            ((p0, p5), (p1, p4), (p2, p3)),
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
        """Returns list of config indices c where counts[c] >= 7"""
        blockers = []
        for k_idx in triple_to_k222s[t_idx]:
            if counts[k_idx] >= 7:
                blockers.append(k_idx)
        return blockers

    def find_swap_partner(t_idx, active, counts):
        """
        Find an active triple t' that can be swapped with t_idx.
        Condition: t' must be in the intersection of all config_triples[c] 
        for c in get_blockers(t_idx, counts).
        """
        blockers = get_blockers(t_idx, counts)
        if not blockers:
            return None  # Should be addable
        
        # Intersection of sets of triple indices
        common = set(config_triples[blockers[0]])
        for c in blockers[1:]:
            common.intersection_update(config_triples[c])
            
        if not common:
            return None
            
        # Check if any triple in 'common' is active
        for t_prime in common:
            if active[t_prime]:
                return t_prime
        return None

    def local_search(active, counts, current_size):
        improved = True
        while improved and time_left() > 2:
            improved = False
            
            # 1. Try to add any edge
            addable_found = False
            for i in range(num_triples):
                if not active[i] and is_addable(i, counts):
                    active[i] = True
                    current_size += 1
                    for k_idx in triple_to_k222s[i]:
                        counts[k_idx] += 1
                    improved = True
                    addable_found = True
                    break  # Restart loop to check for more adds
            
            if addable_found:
                continue

            # 2. If no addable, try swaps
            inactive_edges = [i for i in range(num_triples) if not active[i]]
            if not inactive_edges:
                break  # Full set (unlikely)
                
            found_swap = False
            for _ in range(30):
                if time_left() < 1:
                    break
                t_idx = random.choice(inactive_edges)
                t_prime = find_swap_partner(t_idx, active, counts)
                if t_prime is not None:
                    # Perform swap
                    active[t_prime] = False
                    for k_idx in triple_to_k222s[t_prime]:
                        counts[k_idx] -= 1
                    
                    active[t_idx] = True
                    for k_idx in triple_to_k222s[t_idx]:
                        counts[k_idx] += 1
                    
                    improved = True
                    found_swap = True
                    break  # Restart local search loop to find adds
                    
            if not found_swap:
                break

        return active, counts, current_size

    def simulated_annealing(active, counts, current_size, max_iters=5000):
        """
        Simulated annealing on the QUBO formulation:
        Energy = -current_size + penalty * (number of K(2,2,2) violations)
        We use a soft penalty approach where we track near-violations.
        """
        T_start = 10.0
        T_end = 0.01
        cooling = (T_end / T_start) ** (1.0 / max_iters)
        
        best_active = active[:]
        best_counts = counts[:]
        best_size = current_size
        best_energy = -current_size  # Lower is better
        
        # Track "soft" violations: count how many configs are at 7 or 8
        # A config at 8 is a hard violation, at 7 is a near-violation
        soft_violations = 0
        for k_idx in range(num_k222s):
            if counts[k_idx] >= 8:
                soft_violations += 10  # Heavy penalty for hard violations
            elif counts[k_idx] == 7:
                soft_violations += 1  # Light penalty for near-violations
        
        T = T_start
        for it in range(max_iters):
            if time_left() < 5:
                break
            
            # Choose a random move: add, remove, or swap
            move_type = random.random()
            if move_type < 0.3:
                # Try to add a random inactive triple
                inactive = [i for i in range(num_triples) if not active[i]]
                if not inactive:
                    continue
                t_idx = random.choice(inactive)
                # Check if addable
                can_add = True
                delta_soft = 0
                for k_idx in triple_to_k222s[t_idx]:
                    if counts[k_idx] == 7:
                        can_add = False
                        break
                    elif counts[k_idx] == 6:
                        delta_soft += 1  # Will go from 6 to 7
                if not can_add:
                    # Compute energy delta for allowing a violation
                    delta_energy = -1 + 10 * (1 if any(counts[k] == 7 for k in triple_to_k222s[t_idx]) else 0)
                    if random.random() < (1.0 / (1.0 + 0.0001) ** (delta_energy / T)):
                        # Accept
                        active[t_idx] = True
                        current_size += 1
                        for k_idx in triple_to_k222s[t_idx]:
                            if counts[k_idx] == 7:
                                soft_violations += 10
                            elif counts[k_idx] == 6:
                                soft_violations += 1
                            counts[k_idx] += 1
                        energy = -current_size + soft_violations
                        if energy < best_energy:
                            best_energy = energy
                            best_size = current_size
                            best_active = active[:]
                            best_counts = counts[:]
            elif move_type < 0.6:
                # Try to remove a random active triple
                active_list = [i for i in range(num_triples) if active[i]]
                if not active_list:
                    continue
                t_idx = random.choice(active_list)
                delta_soft = 0
                for k_idx in triple_to_k222s[t_idx]:
                    if counts[k_idx] == 8:
                        delta_soft -= 10
                    elif counts[k_idx] == 7:
                        delta_soft -= 1
                delta_energy = 1 + delta_soft
                if random.random() < (1.0 / (1.0 + 0.0001) ** (delta_energy / T)):
                    active[t_idx] = False
                    current_size -= 1
                    for k_idx in triple_to_k222s[t_idx]:
                        if counts[k_idx] == 8:
                            soft_violations -= 10
                        elif counts[k_idx] == 7:
                            soft_violations -= 1
                        counts[k_idx] -= 1
                    energy = -current_size + soft_violations
                    if energy < best_energy:
                        best_energy = energy
                        best_size = current_size
                        best_active = active[:]
                        best_counts = counts[:]
            else:
                # Swap: remove one active, add one inactive
                active_list = [i for i in range(num_triples) if active[i]]
                inactive = [i for i in range(num_triples) if not active[i]]
                if not active_list or not inactive:
                    continue
                t_remove = random.choice(active_list)
                t_add = random.choice(inactive)
                
                # Compute energy delta
                delta_energy = 0
                delta_soft = 0
                # Remove t_remove
                for k_idx in triple_to_k222s[t_remove]:
                    if counts[k_idx] == 8:
                        delta_soft -= 10
                    elif counts[k_idx] == 7:
                        delta_soft -= 1
                # Add t_add
                for k_idx in triple_to_k222s[t_add]:
                    if counts[k_idx] == 7:
                        delta_soft += 10
                    elif counts[k_idx] == 6:
                        delta_soft += 1
                delta_energy = delta_soft  # Size stays same
                
                if random.random() < (1.0 / (1.0 + 0.0001) ** (delta_energy / T)):
                    # Perform swap
                    active[t_remove] = False
                    for k_idx in triple_to_k222s[t_remove]:
                        if counts[k_idx] == 8:
                            soft_violations -= 10
                        elif counts[k_idx] == 7:
                            soft_violations -= 1
                        counts[k_idx] -= 1
                    
                    active[t_add] = True
                    for k_idx in triple_to_k222s[t_add]:
                        if counts[k_idx] == 7:
                            soft_violations += 10
                        elif counts[k_idx] == 6:
                            soft_violations += 1
                        counts[k_idx] += 1
                    
                    energy = -current_size + soft_violations
                    if energy < best_energy:
                        best_energy = energy
                        best_size = current_size
                        best_active = active[:]
                        best_counts = counts[:]
            
            T *= cooling
        
        # Return the best valid solution found
        # Filter out any hard violations
        valid_active = [False] * num_triples
        valid_counts = [0] * num_k222s
        valid_size = 0
        for i in range(num_triples):
            if best_active[i]:
                # Check if adding this would cause a violation
                can_add = True
                for k_idx in triple_to_k222s[i]:
                    if valid_counts[k_idx] == 7:
                        can_add = False
                        break
                if can_add:
                    valid_active[i] = True
                    valid_size += 1
                    for k_idx in triple_to_k222s[i]:
                        valid_counts[k_idx] += 1
        
        return valid_active, valid_counts, valid_size

    best_size = 0
    best_active = None
    best_counts = None
    
    # Phase 1: Greedy + Local Search
    active, counts, current_size = greedy_init()
    active, counts, current_size = local_search(active, counts, current_size)
    
    if current_size > best_size:
        best_size = current_size
        best_active = active[:]
        best_counts = counts[:]
        
    # Phase 2: Iterated Local Search with Simulated Annealing
    while time_left() > 10:
        # Perturb
        k = random.randint(5, 20)
        active_edges = [i for i in range(num_triples) if active[i]]
        if len(active_edges) < k:
            k = len(active_edges)
            
        for _ in range(k):
            if not active_edges:
                break
            t_idx = random.choice(active_edges)
            active_edges.remove(t_idx)
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
        
        # Try simulated annealing from current state
        if time_left() > 15:
            sa_active, sa_counts, sa_size = simulated_annealing(active, counts, current_size, max_iters=3000)
            if sa_size > best_size:
                best_size = sa_size
                best_active = sa_active
                best_counts = sa_counts
                active, counts, current_size = sa_active, sa_counts, sa_size
                # Continue local search from improved state
                active, counts, current_size = local_search(active, counts, current_size)
                if current_size > best_size:
                    best_size = current_size
                    best_active = active[:]
                    best_counts = counts[:]

    # Final polish
    if best_active:
        active = best_active[:]
        counts = best_counts[:]
        current_size = best_size
        active, counts, current_size = local_search(active, counts, current_size)
        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            best_counts = counts[:]

    # Construct result from best_active
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
