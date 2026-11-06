import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 118
    
    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_id = {t: i for i, t in enumerate(triples)}
    
    # Precompute K(2,2,2)s
    k222_triples = []
    triple_to_k222s = [[] for _ in range(num_triples)]
    
    def get_partitions(vertices):
        v = list(vertices)
        partitions = []
        for i in range(1, 6):
            pair1 = (v[0], v[i])
            remaining = [v[j] for j in range(1, 6) if j != i]
            for j in range(1, 4):
                pair2 = (remaining[0], remaining[j])
                if j == 1:
                    pair3 = (remaining[2], remaining[3])
                elif j == 2:
                    pair3 = (remaining[1], remaining[3])
                else:
                    pair3 = (remaining[1], remaining[2])
                partitions.append([pair1, pair2, pair3])
        return partitions

    for six_subset in combinations(range(n), 6):
        for partition in get_partitions(six_subset):
            t_ids = []
            for x in partition[0]:
                for y in partition[1]:
                    for z in partition[2]:
                        t = tuple(sorted((x, y, z)))
                        t_id = triple_to_id[t]
                        t_ids.append(t_id)
            
            k_idx = len(k222_triples)
            k222_triples.append(t_ids)
            
            for t_id in t_ids:
                triple_to_k222s[t_id].append(k_idx)
                
    num_k222s = len(k222_triples)
    
    # Precompute: for each k222, which triples are in it (as a set for fast lookup)
    k222_to_triples = [set(t_ids) for t_ids in k222_triples]
    
    # Precompute: for each triple, which other triples share a k222 with it
    # triple_neighbors[t_id] = set of t_ids that share at least one k222
    triple_neighbors = [set() for _ in range(num_triples)]
    for k_idx, t_ids in enumerate(k222_triples):
        t_list = list(t_ids)
        for i in range(len(t_list)):
            for j in range(i+1, len(t_list)):
                triple_neighbors[t_list[i]].add(t_list[j])
                triple_neighbors[t_list[j]].add(t_list[i])
    
    class Solver:
        def __init__(self):
            self.present = set()
            self.k_counts = [0] * num_k222s
            
        def can_add(self, t_id):
            if t_id in self.present:
                return True
            for k_idx in triple_to_k222s[t_id]:
                if self.k_counts[k_idx] == 7:
                    return False
            return True
            
        def add(self, t_id):
            if not self.can_add(t_id):
                return False
            self.present.add(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] += 1
            return True
            
        def remove(self, t_id):
            if t_id not in self.present:
                return
            self.present.remove(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] -= 1
                
        def size(self):
            return len(self.present)

    best_size = 0
    best_set = set()
    
    # Helper to polish the solution (add any missing triple that fits)
    # Uses a dirty-set approach: after adding/removing a triple, only check
    # triples that might have become addable (neighbors in the k222 graph)
    def polish(solver, max_iterations=10):
        # Start with all triples not in present as candidates
        candidates = [t_id for t_id in range(num_triples) if t_id not in solver.present]
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < DEADLINE:
            improved = False
            iterations += 1
            # Only check candidates that might be addable
            new_candidates = []
            for t_id in candidates:
                if t_id not in solver.present:
                    if solver.can_add(t_id):
                        solver.present.add(t_id)
                        for k_idx in triple_to_k222s[t_id]:
                            solver.k_counts[k_idx] += 1
                        improved = True
                        # Mark neighbors as potentially addable (they might now be blocked,
                        # but we'll re-check them)
                        for nb in triple_neighbors[t_id]:
                            if nb not in solver.present:
                                new_candidates.append(nb)
            # Rebuild candidate list: keep old candidates that are still not present,
            # plus newly dirtied ones
            present_set = solver.present
            candidates = [t for t in candidates if t not in present_set]
            # Add new candidates (avoid duplicates)
            existing = set(candidates)
            for t in new_candidates:
                if t not in existing:
                    candidates.append(t)
                    existing.add(t)
            if time.time() > DEADLINE:
                return

    # Helper to do a random valid 1-swap with conflict-aware selection
    def random_swap(solver, rng):
        if not solver.present:
            return False
        
        present_list = list(solver.present)
        
        # Prefer removing triples that are in "tight" K(2,2,2)s (count == 7)
        tight_triples = []
        for t_id in present_list:
            for k_idx in triple_to_k222s[t_id]:
                if solver.k_counts[k_idx] == 7:
                    tight_triples.append(t_id)
                    break
        
        if tight_triples and rng.random() < 0.7:
            t_out = rng.choice(tight_triples)
        else:
            t_out = rng.choice(present_list)
        
        # Try to find a triple that can be added after removing t_out
        solver.remove(t_out)
        
        # Find candidates that can now be added - only check neighbors of t_out
        # plus a random sample of non-present triples
        candidates = []
        # Check all neighbors of t_out (they share a k222, so might be addable)
        for nb in triple_neighbors[t_out]:
            if nb not in solver.present:
                if solver.can_add(nb):
                    candidates.append(nb)
        
        # Also check a random sample of non-present triples
        if len(candidates) < 5:
            non_present = [t for t in range(num_triples) if t not in solver.present and t not in set(candidates)]
            sample_size = min(20, len(non_present))
            if sample_size > 0:
                sample = rng.sample(non_present, sample_size)
                for t_id in sample:
                    if solver.can_add(t_id):
                        candidates.append(t_id)
        
        if not candidates:
            solver.add(t_out)
            return False
        
        # Prefer candidates that unblock the most K(2,2,2)s
        if len(candidates) > 1:
            # Score each candidate by how many K(2,2,2)s it participates in
            scored = []
            for t_id in candidates:
                score = len(triple_to_k222s[t_id])
                scored.append((score, t_id))
            scored.sort(reverse=True)
            # Pick from top candidates with some randomness
            top_k = min(5, len(scored))
            t_in = scored[rng.randint(0, top_k-1)][1]
        else:
            t_in = candidates[0]
        
        if solver.add(t_in):
            return True
        else:
            solver.add(t_out)
            return False

    # Helper to do a random valid 2-swap (remove 2, add 2)
    def random_2swap(solver, rng):
        if len(solver.present) < 2:
            return False
        
        present_list = list(solver.present)
        
        # Prefer removing pairs of triples that are in tight K(2,2,2)s
        tight_triples = []
        for t_id in present_list:
            for k_idx in triple_to_k222s[t_id]:
                if solver.k_counts[k_idx] == 7:
                    tight_triples.append(t_id)
                    break
        
        if len(tight_triples) >= 2 and rng.random() < 0.6:
            t_out1, t_out2 = rng.sample(tight_triples, 2)
        elif len(tight_triples) >= 1 and rng.random() < 0.8:
            t_out1 = rng.choice(tight_triples)
            t_out2 = rng.choice([t for t in present_list if t != t_out1])
        else:
            t_out1, t_out2 = rng.sample(present_list, 2)
        
        solver.remove(t_out1)
        solver.remove(t_out2)
        
        # Find candidates that can now be added - check neighbors of both removed triples
        candidates = []
        neighbor_set = set()
        for nb in triple_neighbors[t_out1]:
            if nb not in solver.present and nb != t_out2 and nb not in neighbor_set:
                neighbor_set.add(nb)
                if solver.can_add(nb):
                    candidates.append(nb)
        for nb in triple_neighbors[t_out2]:
            if nb not in solver.present and nb != t_out1 and nb not in neighbor_set:
                neighbor_set.add(nb)
                if solver.can_add(nb):
                    candidates.append(nb)
        
        # Also check a random sample
        if len(candidates) < 4:
            non_present = [t for t in range(num_triples) if t not in solver.present and t not in neighbor_set]
            sample_size = min(15, len(non_present))
            if sample_size > 0:
                sample = rng.sample(non_present, sample_size)
                for t_id in sample:
                    if solver.can_add(t_id):
                        candidates.append(t_id)
        
        if len(candidates) < 2:
            solver.add(t_out1)
            solver.add(t_out2)
            return False
        
        # Score candidates by participation in K(2,2,2)s
        scored = []
        for t_id in candidates:
            score = len(triple_to_k222s[t_id])
            scored.append((score, t_id))
        scored.sort(reverse=True)
        
        # Try to pick two that don't conflict
        top_k = min(10, len(scored))
        t_in1 = scored[rng.randint(0, top_k-1)][1]
        
        if solver.add(t_in1):
            # Find best second candidate
            remaining = [t_id for _, t_id in scored if t_id != t_in1]
            if remaining:
                t_in2 = remaining[rng.randint(0, min(5, len(remaining))-1)]
                if solver.add(t_in2):
                    return True
                else:
                    solver.remove(t_in1)
                    solver.add(t_out1)
                    solver.add(t_out2)
                    return False
            else:
                solver.remove(t_in1)
                solver.add(t_out1)
                solver.add(t_out2)
                return False
        else:
            solver.add(t_out1)
            solver.add(t_out2)
            return False

    # Helper for aggressive local search: try to improve by removing one and adding multiple
    def aggressive_improve(solver, rng, max_attempts=50):
        if not solver.present:
            return False
        
        improved = False
        for _ in range(max_attempts):
            if time.time() > DEADLINE:
                break
            
            present_list = list(solver.present)
            if not present_list:
                break
            
            # Prefer removing triples from tight K(2,2,2)s
            tight_triples = []
            for t_id in present_list:
                for k_idx in triple_to_k222s[t_id]:
                    if solver.k_counts[k_idx] == 7:
                        tight_triples.append(t_id)
                        break
            
            if tight_triples and rng.random() < 0.7:
                t_out = rng.choice(tight_triples)
            else:
                t_out = rng.choice(present_list)
            
            solver.remove(t_out)
            
            # Find all triples that can now be added - check neighbors first
            addable = []
            for nb in triple_neighbors[t_out]:
                if nb not in solver.present:
                    if solver.can_add(nb):
                        addable.append(nb)
            
            # Add a random sample if needed
            if len(addable) < 3:
                non_present = [t for t in range(num_triples) if t not in solver.present and t not in set(addable)]
                sample_size = min(10, len(non_present))
                if sample_size > 0:
                    sample = rng.sample(non_present, sample_size)
                    for t_id in sample:
                        if solver.can_add(t_id):
                            addable.append(t_id)
            
            if len(addable) >= 2:
                # Score addable triples
                scored = [(len(triple_to_k222s[t_id]), t_id) for t_id in addable]
                scored.sort(reverse=True)
                
                # Try adding pairs from top candidates
                found = False
                for i in range(min(15, len(scored))):
                    t_in1 = scored[i][1]
                    ok1 = solver.add(t_in1)
                    if not ok1:
                        continue
                    
                    for j in range(min(15, len(scored))):
                        if j == i:
                            continue
                        t_in2 = scored[j][1]
                        if solver.add(t_in2):
                            found = True
                            break
                        else:
                            solver.remove(t_in2)
                    
                    if found:
                        improved = True
                        break
                    else:
                        solver.remove(t_in1)
                
                if not found:
                    solver.add(t_out)
            else:
                solver.add(t_out)
        
        return improved

    # Multiple restarts with different random seeds
    # Use more restarts with shorter durations for better exploration
    num_restarts = 30
    time_per_restart = 115.0 / num_restarts
    
    for seed in range(num_restarts):
        if time.time() > DEADLINE:
            break
        
        rng = random.Random(seed * 7919 + 1)
        solver = Solver()
        
        # Greedy Init with random order
        order = list(range(num_triples))
        rng.shuffle(order)
        for t_id in order:
            if time.time() > DEADLINE:
                break
            solver.add(t_id)
            
        # Polish
        polish(solver)
        
        if solver.size() > best_size:
            best_size = solver.size()
            best_set = set(solver.present)
        
        # Simulated Annealing / Random Walk
        # Run for allocated time
        restart_deadline = start_time + (seed + 1) * time_per_restart + 2
        
        while time.time() < restart_deadline and time.time() < DEADLINE:
            # Perform a mix of 1-swaps and 2-swaps
            for _ in range(25):
                if time.time() > restart_deadline or time.time() > DEADLINE:
                    break
                if rng.random() < 0.5:
                    random_swap(solver, rng)
                else:
                    random_2swap(solver, rng)
            
            # Polish after each batch
            polish(solver, max_iterations=5)
            
            # Occasionally do aggressive improvement
            if rng.random() < 0.3:
                aggressive_improve(solver, rng, max_attempts=30)
            
            if solver.size() > best_size:
                best_size = solver.size()
                best_set = set(solver.present)
        
        # Final polish for this restart
        polish(solver, max_iterations=15)
        
        if solver.size() > best_size:
            best_size = solver.size()
            best_set = set(solver.present)
    
    # Final high-precision polish pass with the best solution
    if best_set:
        final_solver = Solver()
        for t_id in best_set:
            final_solver.present.add(t_id)
            for k_idx in triple_to_k222s[t_id]:
                final_solver.k_counts[k_idx] += 1
        
        # Aggressive final polish with more iterations
        polish(final_solver, max_iterations=100)
        
        # Try additional swaps on the final solution
        final_rng = random.Random(12345)
        final_deadline = DEADLINE - 2
        while time.time() < final_deadline:
            if final_rng.random() < 0.5:
                random_swap(final_solver, final_rng)
            else:
                random_2swap(final_solver, final_rng)
            polish(final_solver, max_iterations=5)
            
            if final_solver.size() > best_size:
                best_size = final_solver.size()
                best_set = set(final_solver.present)
        
        # One more final polish
        polish(final_solver, max_iterations=50)
        
        if final_solver.size() > best_size:
            best_size = final_solver.size()
            best_set = set(final_solver.present)
    
    return [list(triples[t_id]) for t_id in best_set]
