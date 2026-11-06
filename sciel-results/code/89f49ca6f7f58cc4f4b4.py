import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
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

    def greedy_init(weights=None):
        """
        Greedy initialization using dynamic weights.
        If weights is None, use random order.
        If weights is a list, sort by weight descending.
        """
        active = [False] * num_triples
        counts = [0] * num_k222s
        current_size = 0
        
        if weights is None:
            order = list(range(num_triples))
            random.shuffle(order)
        else:
            # Sort indices by weight descending, break ties randomly
            # To avoid stable sort bias, add small random noise
            noisy_weights = [(weights[i], random.random()) for i in range(num_triples)]
            noisy_weights.sort(key=lambda x: x[0], reverse=True)
            order = [i for i, _ in noisy_weights]
        
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
            return None # Should be addable
        
        # Intersection of sets of triple indices
        # Start with the set of triples in the first blocker config
        # Note: config_triples[c] is a list of triple indices
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
                break
                
            # Try a few random swaps
            found_swap = False
            # Increase attempts slightly to help escape
            for _ in range(100): 
                if time_left() < 1: break
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
                    break
                    
            if not found_swap:
                pass

        return active, counts, current_size

    best_size = 0
    best_active = None
    best_counts = None
    
    # Weight tracking for dynamic greedy
    # weights[i] represents how "good" or "central" triple i is.
    # We can update weights based on how often a triple was part of a blocking configuration
    # or simply use degree in the conflict graph as a static weight initially.
    # Let's try a simple dynamic weight: start with 0, increase weight if it was removed in perturbation?
    # Or better: use the "conflict potential".
    
    # Initialize weights to 0
    weights = [0.0] * num_triples
    
    # Run multiple rounds of ILS with weight updates
    num_rounds = 50
    for r in range(num_rounds):
        if time_left() < 10:
            break
            
        # 1. Greedy Init with current weights
        active, counts, current_size = greedy_init(weights)
        
        # 2. Local Search
        active, counts, current_size = local_search(active, counts, current_size)
        
        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            best_counts = counts[:]
            
        # 3. Update Weights
        # Heuristic: Increase weight of triples that were active in the best solution but not in current?
        # Or: Penalize triples that are in many blocking configs.
        # Let's try: For each config that is "full" (count 7) in the current solution,
        # the 8th triple (or any triple that would block) is "bad"?
        # Actually, let's look at the triples that were REMOVED during local search swaps.
        # We didn't track those easily.
        
        # Alternative: Update weights based on the "density" of conflicts.
        # For each triple i, weight[i] = number of configs it belongs to where the config is "tight" (count >= 6)?
        # This is expensive to recompute every round.
        
        # Simpler: Randomly perturb weights? No, that's just random restart.
        
        # Let's try: Weight[i] += 1 if triple i was in the best solution, else 0.
        # This biases future greedy runs towards the best found solution structure.
        for i in range(num_triples):
            if best_active and best_active[i]:
                weights[i] += 0.1
            else:
                weights[i] -= 0.05
        
        # Normalize weights to avoid drift
        max_w = max(weights) if weights else 0
        if max_w > 0:
            for i in range(num_triples):
                weights[i] /= max_w

        # 4. Perturb best solution for next round's start?
        # The current loop restarts greedy from scratch each time.
        # To make it true ILS, we should perturb the *current* best and run local search.
        # Let's do that instead of restarting greedy.
        
        # Reset loop logic: 
        # Actually, the above loop is "Weighted Greedy + LS".
        # Let's add a second phase: Perturb Best + LS.
        
    # Phase 2: Iterated Local Search on the best solution found
    while time_left() > 5:
        # Perturb best
        active = best_active[:]
        counts = best_counts[:]
        current_size = best_size
        
        active_edges = [i for i in range(num_triples) if active[i]]
        
        # Random removal
        k = random.randint(5, 20)
        for _ in range(k):
            if not active_edges: break
            t_idx = random.choice(active_edges)
            active_edges.remove(t_idx)
            active[t_idx] = False
            current_size -= 1
            for k_idx in triple_to_k222s[t_idx]:
                counts[k_idx] -= 1
                
        # Local Search
        active, counts, current_size = local_search(active, counts, current_size)
        
        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            best_counts = counts[:]

    # Construct result
    if best_active is None:
        active, counts, current_size = greedy_init(None)
        best_active = active
        
    result = []
    for i in range(num_triples):
        if best_active[i]:
            result.append(list(all_triples[i]))
            
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Number of triples: {len(result)}")
