import itertools
import random
import math
import time

def construct(v, k, t):
    # Parameters
    total_subsets = list(itertools.combinations(range(v), t))
    subset_index = {s: i for i, s in enumerate(total_subsets)}
    n_subsets = len(total_subsets)
    
    # Greedy Initialization
    uncovered = set(range(n_subsets))
    blocks = []
    
    # Helper to get block from set of points
    def get_block(points):
        return tuple(sorted(points))

    # Greedy Step
    while uncovered:
        # Pick a random uncovered subset to target
        target_idx = random.choice(list(uncovered))
        target = total_subsets[target_idx]
        
        # Start block with target
        block = list(target)
        
        # Greedily add points
        while len(block) < k:
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Calculate gain
                # A point x covers an uncovered subset s if s is subset of block U {x}
                # and s contains at least one element from block?
                # Actually, s is covered by block U {x} if s is subset of block U {x}.
                # Since s has size t, and block has size < k.
                # If s is uncovered, it's not in any existing block.
                # We want to cover as many uncovered subsets as possible.
                # A subset s is covered by block U {x} if s is a subset of block U {x}.
                # This implies s is a subset of (block U {x}).
                # Since s has size t, and we are adding x, s must be a subset of (block U {x}).
                # This is equivalent to: s is a subset of block U {x}.
                
                # Efficient check:
                # For each uncovered subset s, check if s is subset of block U {x}.
                # This is slow if we iterate all uncovered.
                # Instead, iterate uncovered subsets that intersect block U {x}?
                
                # Let's just iterate all uncovered subsets for simplicity, v=24 is small.
                gain = 0
                for s_idx in uncovered:
                    s = total_subsets[s_idx]
                    # Check if s is subset of block U {x}
                    # s is subset of block U {x} iff (s - {x}) is subset of block
                    # and x is in s OR s is subset of block (but s is uncovered, so s not subset of block)
                    # So s must contain x? Not necessarily.
                    # Wait, if s does not contain x, then s must be subset of block.
                    # But s is uncovered, so s is not subset of any previous block.
                    # If s is subset of current partial block, it would have been covered?
                    # No, partial block is not a final block yet.
                    # However, if s is subset of block, then adding x doesn't help cover s,
                    # s is already "covered" by the partial block? No, the partial block isn't added yet.
                    # But if s is subset of block, then when we finish the block, s will be covered.
                    # So gain should count s if s is subset of (block U {x}).
                    
                    # Optimization:
                    # If x not in s: s must be subset of block.
                    # If x in s: s - {x} must be subset of block.
                    
                    if x in s:
                        if all(p in block for p in s if p != x):
                            gain += 1
                    else:
                        if all(p in block for p in s):
                            gain += 1
                            
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point != -1:
                block.append(best_point)
            else:
                # Fallback: pick random
                available = [x for x in range(v) if x not in block]
                if available:
                    block.append(random.choice(available))
                else:
                    break

        block = tuple(sorted(block))
        if block not in blocks:
            blocks.append(block)
            # Update uncovered
            for s in itertools.combinations(block, t):
                idx = subset_index[s]
                if idx in uncovered:
                    uncovered.discard(idx)

    # Simulated Annealing
    # State: blocks (list of tuples)
    # Cost: uncovered_count * 10000 + len(blocks)
    
    def calculate_cost(blocks):
        covered = [0] * n_subsets
        for b in blocks:
            for s in itertools.combinations(b, t):
                covered[subset_index[s]] += 1
        unc = sum(1 for c in covered if c == 0)
        return unc * 10000 + len(blocks)
    
    current_blocks = list(blocks)
    current_cost = calculate_cost(current_blocks)
    
    start_time = time.time()
    time_limit = 110 # Leave 10s buffer
    T = 100.0
    cooling = 0.999
    
    best_blocks = list(current_blocks)
    best_cost = current_cost
    
    while time.time() - start_time < time_limit:
        # Generate move
        move_type = random.random()
        new_blocks = list(current_blocks)
        
        if move_type < 0.3:
            # Remove a block
            if new_blocks:
                idx = random.randint(0, len(new_blocks) - 1)
                new_blocks.pop(idx)
        elif move_type < 0.6:
            # Add a block
            # Pick a random uncovered subset if any, else random subset
            covered = [0] * n_subsets
            for b in new_blocks:
                for s in itertools.combinations(b, t):
                    covered[subset_index[s]] += 1
            unc = [i for i, c in enumerate(covered) if c == 0]
            if unc:
                target_idx = random.choice(unc)
                target = total_subsets[target_idx]
            else:
                target_idx = random.randint(0, n_subsets - 1)
                target = total_subsets[target_idx]
            
            block = list(target)
            # Fill rest randomly
            remaining = [x for x in range(v) if x not in block]
            random.shuffle(remaining)
            block.extend(remaining[:k-len(block)])
            new_block = tuple(sorted(block))
            if new_block not in new_blocks:
                new_blocks.append(new_block)
        else:
            # Swap a point in a random block
            if new_blocks:
                b_idx = random.randint(0, len(new_blocks) - 1)
                b = list(new_blocks[b_idx])
                p = random.choice(b)
                outside = [x for x in range(v) if x not in b]
                if outside:
                    q = random.choice(outside)
                    b.remove(p)
                    b.append(q)
                    new_block = tuple(sorted(b))
                    new_blocks[b_idx] = new_block

        new_cost = calculate_cost(new_blocks)
        
        # Acceptance
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_blocks = new_blocks
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_blocks = list(current_blocks)
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    return [list(b) for b in best_blocks]
