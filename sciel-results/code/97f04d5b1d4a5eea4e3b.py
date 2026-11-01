import itertools
import random
import time

def construct(v, k, t):
    # Precomputation
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    
    # Map triple to index
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # block_to_triples: list of triple indices for each block
    block_to_triples = []
    for b in all_blocks:
        ts = [triple_to_idx[tr] for tr in itertools.combinations(b, t)]
        block_to_triples.append(ts)
        
    # triple_to_blocks: list of block indices for each triple
    triple_to_blocks = [[] for _ in range(num_triples)]
    for i, ts in enumerate(block_to_triples):
        for t_idx in ts:
            triple_to_blocks[t_idx].append(i)

    # Local Search Solver
    # We want to minimize: num_uncovered + alpha * num_selected
    # Ideally num_uncovered = 0 and num_selected is minimal.
    
    def local_search(start_blocks, max_time, alpha=100):
        # start_blocks: set of block indices
        x = [0] * num_blocks
        for b in start_blocks:
            x[b] = 1
            
        # Initialize state
        triple_cover_count = [0] * num_triples
        for b in start_blocks:
            for t_idx in block_to_triples[b]:
                triple_cover_count[t_idx] += 1
                
        uncovered = set()
        for i in range(num_triples):
            if triple_cover_count[i] == 0:
                uncovered.add(i)
                
        current_selected = len(start_blocks)
        current_uncovered = len(uncovered)
        current_cost = current_uncovered * alpha + current_selected
        
        best_selected = current_selected if current_uncovered == 0 else float('inf')
        best_solution = set(start_blocks) if current_uncovered == 0 else None
        
        start_time = time.time()
        
        while time.time() - start_time < max_time:
            if current_uncovered == 0:
                # We have a valid solution. Try to minimize size.
                if current_selected < best_selected:
                    best_selected = current_selected
                    best_solution = set(start_blocks)
                    
            # Pick a random uncovered triple if any, else pick a random selected block to try to remove
            if current_uncovered > 0:
                # Pick random uncovered triple
                t_idx = random.choice(list(uncovered))
                # Candidates are blocks covering this triple
                candidates = triple_to_blocks[t_idx]
            else:
                # No uncovered triples. Try to remove a block.
                # Candidates are selected blocks
                candidates = [b for b in start_blocks]
                if not candidates:
                    break
                
            # Evaluate flips for a subset of candidates to save time
            # Shuffle and take first N
            random.shuffle(candidates)
            num_eval = min(len(candidates), 50)
            
            best_delta = float('inf')
            best_flip = None
            best_new_cost = float('inf')
            
            for b in candidates[:num_eval]:
                if x[b] == 0:
                    # Flip 0 -> 1
                    # Cost change: +1 (block) - (newly covered triples) * alpha
                    newly_covered = 0
                    for t_idx in block_to_triples[b]:
                        if triple_cover_count[t_idx] == 0:
                            newly_covered += 1
                    delta_cost = 1 - newly_covered * alpha
                else:
                    # Flip 1 -> 0
                    # Cost change: -1 (block) + (newly uncovered triples) * alpha
                    newly_uncovered = 0
                    for t_idx in block_to_triples[b]:
                        if triple_cover_count[t_idx] == 1:
                            newly_uncovered += 1
                    delta_cost = -1 + newly_uncovered * alpha
                
                if delta_cost < best_new_cost:
                    best_new_cost = delta_cost
                    best_flip = b
                    best_delta = delta_cost
            
            if best_flip is None:
                # No candidates evaluated or something wrong
                break
                
            # Apply flip with some probability (to allow local search noise)
            # If delta is negative, always accept. If positive, accept with prob p.
            if best_delta < 0:
                accept = True
            else:
                # Simple annealing-like acceptance
                accept = random.random() < 0.1 
            
            if accept:
                b = best_flip
                if x[b] == 0:
                    x[b] = 1
                    start_blocks.add(b)
                    current_selected += 1
                    for t_idx in block_to_triples[b]:
                        if triple_cover_count[t_idx] == 0:
                            uncovered.discard(t_idx)
                            current_uncovered -= 1
                        triple_cover_count[t_idx] += 1
                else:
                    x[b] = 0
                    start_blocks.discard(b)
                    current_selected -= 1
                    for t_idx in block_to_triples[b]:
                        triple_cover_count[t_idx] -= 1
                        if triple_cover_count[t_idx] == 0:
                            uncovered.add(t_idx)
                            current_uncovered += 1
                current_cost += best_delta
            else:
                # If we didn't accept, and we are stuck, maybe restart?
                # For now, just continue.
                pass
                
        return best_solution, best_selected

    # Strategy:
    # 1. Get a quick valid solution using Greedy (fast).
    # 2. Use Local Search to improve it.
    
    # Quick Greedy for initialization
    def quick_greedy():
        uncovered = set(range(num_triples))
        gains = [len(ts) for ts in block_to_triples]
        blocks_used = set()
        
        while uncovered:
            max_gain = -1
            best_b = -1
            # Optimize max finding: just pick random among top few?
            # For init, simple max is fine.
            for i in range(num_blocks):
                if gains[i] > max_gain:
                    max_gain = gains[i]
                    best_b = i
            
            if max_gain == 0:
                break
                
            blocks_used.add(best_b)
            
            for t_idx in block_to_triples[best_b]:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
                    for b_prime in triple_to_blocks[t_idx]:
                        gains[b_prime] -= 1
                        
        return blocks_used

    init_solution = quick_greedy()
    
    # Run Local Search
    # We have ~100s.
    # Split time into chunks for restarts with different initializations?
    # Or just run one long local search from the greedy start.
    # Local search might get stuck.
    # Let's do a few restarts with perturbed greedy starts.
    
    best_overall = float('inf')
    best_sol_overall = None
    
    total_time = 110
    num_restarts = 10
    time_per_restart = total_time / num_restarts
    
    for r in range(num_restarts):
        # Perturb the initial solution
        if r == 0:
            init = init_solution
        else:
            # Remove some random blocks and let local search fix it?
            # Or just use the previous best solution perturbed.
            if best_sol_overall is not None:
                init = set(best_sol_overall)
                # Remove 1-5 random blocks to kick it out of local min
                to_remove = random.sample(list(init), min(3, len(init)))
                for b in to_remove:
                    init.discard(b)
            else:
                init = init_solution
        
        sol, size = local_search(init, time_per_restart, alpha=50)
        
        if sol is not None and size < best_overall:
            best_overall = size
            best_sol_overall = sol
            
        if time.time() > time.time() + total_time - 5: # Safety check
            break

    if best_sol_overall is None:
        # Fallback
        best_sol_overall = init_solution
        best_overall = len(best_sol_overall)
        
    return [list(all_blocks[b]) for b in best_sol_overall]
