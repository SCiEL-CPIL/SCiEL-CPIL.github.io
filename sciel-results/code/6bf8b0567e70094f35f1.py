import itertools
import time
import random

def construct(v, k, t):
    # Precompute
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    # Map subset to index for faster counting? 
    # Or just use set of tuples.
    
    # Deadline
    start_time = time.time()
    deadline = start_time + 110 # 10s buffer
    
    best_blocks = None
    best_len = float('inf')
    
    # Precompute combinations of sizes 1..k-1 for gain calculation?
    # Gain for adding x to B: count S in uncovered such that S subset B U {x} and x in S.
    # S \ {x} is a (t-1)-subset of B.
    
    while time.time() < deadline:
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick random uncovered subset
            s0 = random.choice(list(uncovered))
            block = list(s0)
            
            # Remove all t-subsets of block from uncovered
            # Since block is size t initially, only s0 is in block?
            # Yes, if len(block) == t.
            # But we start with s0 which is size t.
            # So remove s0.
            uncovered.difference_update(itertools.combinations(block, t))
            
            while len(block) < k:
                best_gain = -1
                best_x = None
                candidates = [x for x in points if x not in block]
                random.shuffle(candidates) # Random tie-breaking
                
                for x in candidates:
                    gain = 0
                    # S must contain x and be subset of block U {x}
                    # So S \ {x} is a (t-1)-subset of block
                    for comb in itertools.combinations(block, t-1):
                        s = tuple(sorted(comb + (x,)))
                        if s in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x is not None:
                        # Random tie break
                        if random.random() < 0.5:
                            best_x = x
                
                if best_x is not None:
                    block.append(best_x)
                    # Update uncovered
                    # Remove all t-subsets of block that contain best_x
                    # These are comb U {best_x} where comb is (t-1)-subset of block \ {best_x}
                    to_remove = []
                    for comb in itertools.combinations(block, t-1):
                        # Check if best_x is in comb? No, comb is from block which includes best_x now.
                        # We want subsets S of block such that best_x in S.
                        # S \ {best_x} is (t-1)-subset of block \ {best_x}.
                        pass # Logic correction needed below
                    
                    # Correct update:
                    # S is a t-subset of block.
                    # S contains best_x.
                    # So S = {best_x} U C, where C is a (t-1)-subset of (block \ {best_x}).
                    block_wo_x = [x for x in block if x != best_x]
                    for comb in itertools.combinations(block_wo_x, t-1):
                        s = tuple(sorted(comb + (best_x,)))
                        to_remove.append(s)
                    uncovered.difference_update(to_remove)
            
            blocks.append(tuple(sorted(block)))
        
        # Post-process: Remove redundant blocks
        blocks = remove_redundant(blocks, all_t_subsets, v, k, t)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    return [list(b) for b in best_blocks]

def remove_redundant(blocks, all_t_subsets, v, k, t):
    # Count coverage
    cov = {s: 0 for s in all_t_subsets}
    for b in blocks:
        for s in itertools.combinations(b, t):
            cov[s] += 1
            
    changed = True
    current_blocks = list(blocks)
    while changed:
        changed = False
        new_blocks = []
        for b in current_blocks:
            redundant = True
            for s in itertools.combinations(b, t):
                if cov[s] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove b
                for s in itertools.combinations(b, t):
                    cov[s] -= 1
                changed = True
            else:
                new_blocks.append(b)
        current_blocks = new_blocks
    return current_blocks
