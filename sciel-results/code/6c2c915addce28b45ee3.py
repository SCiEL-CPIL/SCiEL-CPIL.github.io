import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    all_tsets = list(itertools.combinations(points, t))
    
    num_blocks = len(all_blocks)
    num_tsets = len(all_tsets)
    
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Pre-compute block_tsets: list of lists of t-set indices
    block_tsets = []
    for b in all_blocks:
        b_set = set(b)
        # Only combinations of b of size t
        covered = [tset_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_tsets.append(covered)
        
    # Pre-compute tset_blocks: list of lists of block indices
    tset_blocks = [[] for _ in range(num_tsets)]
    for b_idx, covered in enumerate(block_tsets):
        for s_idx in covered:
            tset_blocks[s_idx].append(b_idx)
            
    # Greedy Initialization
    cover_count = [0] * num_tsets
    uncovered_set = set(range(num_tsets))
    solution = set()
    
    # Greedy: Pick block covering most uncovered
    # To speed up, we can use a heuristic:
    # While uncovered_set:
    #   Pick s in uncovered_set with min len(tset_blocks[s])
    #   Pick b in tset_blocks[s] maximizing |block_tsets[b] \cap uncovered_set|
    
    while uncovered_set and time.time() < deadline - 5:
        # Find most constrained s
        min_deg = float('inf')
        target_s = None
        # Optimization: If we have many, sampling might be better, 
        # but 11k is small enough to scan.
        for s in uncovered_set:
            deg = len(tset_blocks[s])
            if deg < min_deg:
                min_deg = deg
                target_s = s
                if min_deg == 1:
                    break
                    
        if target_s is None:
            break
            
        best_b = -1
        max_gain = -1
        for b_idx in tset_blocks[target_s]:
            if b_idx in solution:
                continue
            gain = 0
            for s2 in block_tsets[b_idx]:
                if s2 in uncovered_set:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_b = b_idx
                
        if best_b != -1:
            solution.add(best_b)
            for s in block_tsets[best_b]:
                cover_count[s] += 1
                if cover_count[s] == 1:
                    uncovered_set.discard(s)
        else:
            # Fallback
            b_idx = tset_blocks[target_s][0]
            solution.add(b_idx)
            for s in block_tsets[b_idx]:
                cover_count[s] += 1
                if cover_count[s] == 1:
                    uncovered_set.discard(s)
                    
    # Ensure valid
    if uncovered_set:
        for s in list(uncovered_set):
            for b_idx in tset_blocks[s]:
                if b_idx not in solution:
                    solution.add(b_idx)
                    for s2 in block_tsets[b_idx]:
                        cover_count[s2] += 1
                        if cover_count[s2] == 1:
                            uncovered_set.discard(s2)
                    break

    best_solution = set(solution)
    best_count = len(solution)
    
    # Simulated Annealing
    LAMBDA = 50.0
    T = 5.0
    COOLING = 0.999
    MIN_T = 0.05
    
    # Helper to add/remove block and update state
    def add_block(b_idx):
        nonlocal cover_count, uncovered_set
        for s in block_tsets[b_idx]:
            cover_count[s] += 1
            if cover_count[s] == 1:
                uncovered_set.discard(s)
                
    def remove_block(b_idx):
        nonlocal cover_count, uncovered_set
        for s in block_tsets[b_idx]:
            cover_count[s] -= 1
            if cover_count[s] == 0:
                uncovered_set.add(s)
                
    while time.time() < deadline:
        if T < MIN_T:
            T = 5.0 # Restart
            
        # 70% Remove, 30% Add
        if random.random() < 0.7:
            if not solution: continue
            b_idx = random.choice(list(solution))
            
            # Calculate delta
            # If we remove b_idx, how many s become uncovered?
            delta_uncovered = 0
            for s in block_tsets[b_idx]:
                if cover_count[s] == 1:
                    delta_uncovered += 1
            
            delta_count = -1
            delta_e = delta_count + LAMBDA * delta_uncovered
            
            if delta_e < 0 or (random.random() < math.exp(-delta_e / T)):
                # Accept
                solution.discard(b_idx)
                remove_block(b_idx)
                
                # Check best
                if not uncovered_set:
                    if len(solution) < best_count:
                        best_count = len(solution)
                        best_solution = set(solution)
        else:
            # Add
            # Pick random block
            b_idx = random.randint(0, num_blocks - 1)
            if b_idx in solution:
                continue
                
            # Calculate delta
            # If we add b_idx, how many s become covered?
            delta_covered = 0
            for s in block_tsets[b_idx]:
                if cover_count[s] == 0:
                    delta_covered += 1
            
            # delta_uncovered = -delta_covered
            delta_count = 1
            delta_e = delta_count + LAMBDA * (-delta_covered)
            
            if delta_e < 0 or (random.random() < math.exp(-delta_e / T)):
                # Accept
                solution.add(b_idx)
                add_block(b_idx)
                
                # Check best
                if not uncovered_set:
                    if len(solution) < best_count:
                        best_count = len(solution)
                        best_solution = set(solution)
                        
        T *= COOLING
        
    return [list(all_blocks[b]) for b in sorted(best_solution)]
