# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    """
    Construct a triple system on 16 points without the complete 3-partite K(2,2,2).
    Uses a Max-SAT style local search with efficient candidate management.
    """
    n = 16
    num_triples = 560  # C(16,3)
    
    # Precompute all possible triples
    all_triples = []
    triple_index = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triple = (i, j, k)
                triple_index[triple] = len(all_triples)
                all_triples.append(triple)
    
    # Precompute all K(2,2,2) configurations
    k222_configs = []
    
    def get_pairings(points):
        points = list(points)
        pairings = []
        def rec(remaining, current):
            if not remaining:
                pairings.append(current[:])
                return
            first = remaining[0]
            rest = remaining[1:]
            for i in range(len(rest)):
                second = rest[i]
                new_remaining = rest[:i] + rest[i+1:]
                rec(new_remaining, current + [(first, second)])
        rec(points, [])
        return pairings
    
    seen_configs = set()
    for six_points in combinations(range(n), 6):
        pairings = get_pairings(six_points)
        for pairing in pairings:
            p1, p2, p3 = pairing
            triples_in_config = []
            for a in p1:
                for b in p2:
                    for c in p3:
                        triple = tuple(sorted((a, b, c)))
                        idx = triple_index[triple]
                        triples_in_config.append(idx)
            config_key = tuple(sorted(triples_in_config))
            if config_key not in seen_configs:
                seen_configs.add(config_key)
                k222_configs.append(triples_in_config)
    
    num_configs = len(k222_configs)
    
    # Precompute for each triple, which configs it belongs to
    triple_to_configs = [[] for _ in range(num_triples)]
    for c_idx, config in enumerate(k222_configs):
        for t_idx in config:
            triple_to_configs[t_idx].append(c_idx)
            
    random.seed(42)
    start_time = time.time()
    time_limit = 115
    
    best_solution = None
    best_count = -1
    
    def greedy_init():
        sol = [False] * num_triples
        counts = [0] * num_configs
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        
        for idx in candidates:
            safe = True
            for c_idx in triple_to_configs[idx]:
                if counts[c_idx] == 7:
                    safe = False
                    break
            if safe:
                sol[idx] = True
                for c_idx in triple_to_configs[idx]:
                    counts[c_idx] += 1
        return sol, counts, sum(sol)
    
    def is_safe(sol, counts, idx):
        if sol[idx]:
            return False
        for c_idx in triple_to_configs[idx]:
            if counts[c_idx] == 7:
                return False
        return True

    def add_triple(sol, counts, idx):
        sol[idx] = True
        for c_idx in triple_to_configs[idx]:
            counts[c_idx] += 1

    def remove_triple(sol, counts, idx):
        sol[idx] = False
        for c_idx in triple_to_configs[idx]:
            counts[c_idx] -= 1

    def local_search(sol, counts, current_count):
        nonlocal best_solution, best_count
        
        stagnation = 0
        max_stagnation = 50
        
        while time.time() - start_time < time_limit:
            # 1. Try to add any safe triple
            added_any = False
            # To speed up, we can keep a list of candidates, but checking all 560 is fast enough
            # Let's pick a random subset or just iterate
            # Iterating all is O(560 * deg) ~ 5600 ops, very fast.
            safe_indices = []
            for i in range(num_triples):
                if is_safe(sol, counts, i):
                    safe_indices.append(i)
            
            if safe_indices:
                # Pick one randomly to avoid bias, or pick one with highest "potential"
                idx = random.choice(safe_indices)
                add_triple(sol, counts, idx)
                current_count += 1
                stagnation = 0
                
                if current_count > best_count:
                    best_count = current_count
                    best_solution = sol[:]
                continue
            
            # 2. No safe triple to add. Must remove one.
            # Find triples that are in tight clauses (count == 7)
            tight_triples = []
            for c_idx, cnt in enumerate(counts):
                if cnt == 7:
                    for t_idx in k222_configs[c_idx]:
                        if sol[t_idx]:
                            tight_triples.append(t_idx)
            
            if not tight_triples:
                # Should not happen if safe_indices is empty and count > 0?
                # If no tight clauses, then all clauses have count < 7, so all triples are safe?
                # No, a triple is unsafe if it completes a clause to 8.
                # If no clause has count 7, then adding any triple keeps count <= 7?
                # Wait. If max count is 6, adding a triple makes it 7. Still safe.
                # So if no clause has count 7, ALL inactive triples are safe.
                # This contradicts safe_indices being empty.
                # So if safe_indices is empty, there MUST be clauses with count 7.
                # Unless num_triples is 0? No.
                break

            # Choose a kick
            # Heuristic: Pick a triple from tight_triples that appears in the most tight clauses?
            # Or just random.
            kick_idx = random.choice(tight_triples)
            
            # Save state to revert if bad
            removed = [kick_idx]
            # Add kick to "pool" of removed, try to add others
            remove_triple(sol, counts, kick_idx)
            current_count -= 1
            
            # Try to add as many as possible
            added = []
            while True:
                # Find safe
                safe_now = []
                for i in range(num_triples):
                    if is_safe(sol, counts, i):
                        safe_now.append(i)
                if not safe_now:
                    break
                # Pick one
                idx = random.choice(safe_now)
                add_triple(sol, counts, idx)
                added.append(idx)
                current_count += 1
            
            # Check if we improved
            if current_count > best_count:
                best_count = current_count
                best_solution = sol[:]
                stagnation = 0
            else:
                # Revert if no improvement in count?
                # Actually, if current_count == previous_count - 1 + len(added)
                # If len(added) == 0, we lost 1. Revert.
                # If len(added) == 1, we are equal. Might be worth keeping to escape local opt?
                # If len(added) > 1, we gained.
                
                if len(added) < 1:
                    # Revert
                    for idx in added:
                        remove_triple(sol, counts, idx)
                        current_count -= 1
                    add_triple(sol, counts, kick_idx)
                    current_count += 1
                    stagnation += 1
                else:
                    # Keep the change, even if count didn't increase strictly, 
                    # but if it did, we updated best.
                    # If count is same, we might be stuck.
                    if current_count <= best_count:
                         stagnation += 1
                    else:
                         stagnation = 0

            if stagnation > max_stagnation:
                break

    # Run multiple restarts
    for restart in range(200):
        if time.time() - start_time > time_limit:
            break
            
        sol, counts, current_count = greedy_init()
        
        if current_count > best_count:
            best_count = current_count
            best_solution = sol[:]
            
        local_search(sol, counts, current_count)
        
        if best_solution is None:
            best_solution, _, _ = greedy_init()
            best_count = sum(best_solution)

    result = [list(all_triples[i]) for i in range(num_triples) if best_solution[i]]
    return result
# EVOLVE-BLOCK-END
