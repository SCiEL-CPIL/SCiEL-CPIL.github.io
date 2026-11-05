import itertools
import time
import random

def construct(v, k, t):
    all_points = range(v)
    
    def bits_to_mask(bits):
        m = 0
        for b in bits:
            m |= (1 << b)
        return m

    def mask_to_points(mask):
        return [i for i in range(v) if (mask >> i) & 1]

    # Pre-compute all t-subset masks
    all_t_masks = []
    for combo in itertools.combinations(all_points, t):
        all_t_masks.append(bits_to_mask(combo))
        
    def run_greedy(seed_val):
        random.seed(seed_val)
        uncovered = set(all_t_masks)
        blocks = []
        coverage_count = {m: 0 for m in all_t_masks}

        def add_block(block_pts):
            nonlocal uncovered, blocks, coverage_count
            blocks.append(sorted(block_pts))
            for sub in itertools.combinations(block_pts, t):
                m = bits_to_mask(sub)
                coverage_count[m] += 1
                if coverage_count[m] == 1:
                    if m in uncovered:
                        uncovered.remove(m)

        def remove_block(block_pts):
            nonlocal uncovered, blocks, coverage_count
            idx = blocks.index(sorted(block_pts))
            blocks.pop(idx)
            for sub in itertools.combinations(block_pts, t):
                m = bits_to_mask(sub)
                coverage_count[m] -= 1
                if coverage_count[m] == 0:
                    uncovered.add(m)

        def prune_blocks():
            nonlocal uncovered, blocks, coverage_count
            changed = True
            while changed:
                changed = False
                for i in range(len(blocks) - 1, -1, -1):
                    b = blocks[i]
                    essential = False
                    for sub in itertools.combinations(b, t):
                        m = bits_to_mask(sub)
                        if coverage_count[m] == 1:
                            essential = True
                            break
                    if not essential:
                        remove_block(b)
                        changed = True

        # Phase 1: Stochastic Greedy Construction
        if k <= v:
            # Randomly select the first block? Or fix it?
            # Fixing it reduces symmetry. Let's pick a random first block from all C(v,k) blocks?
            # No, that's expensive to generate.
            # Let's pick a random t-subset as the start, then expand.
            # The seed used min(uncovered). Let's stick to min(uncovered) for the *first* block to ensure coverage of "small" indices early, 
            # but randomize the expansion.
            
            # Actually, to break symmetry more effectively, let's pick a random uncovered t-subset to start the first block.
            start_mask = random.choice(list(uncovered)) if uncovered else None
            if start_mask is not None:
                start_pts = mask_to_points(start_mask)
                block_pts = start_pts[:]
                
                while len(block_pts) < k:
                    best_gain = -1
                    candidates = []
                    
                    t_minus_1_subs = list(itertools.combinations(block_pts, t-1))
                    
                    for x in range(v):
                        if x in block_pts:
                            continue
                        
                        gain = 0
                        for sub in t_minus_1_subs:
                            m = 0
                            for p in sub:
                                m |= (1 << p)
                            m |= (1 << x)
                            if m in uncovered:
                                gain += 1
                        
                        if gain > best_gain:
                            best_gain = gain
                            candidates = [x]
                        elif gain == best_gain:
                            candidates.append(x)
                    
                    if candidates:
                        best_x = random.choice(candidates)
                    else:
                        # Fallback: pick smallest available
                        for x in range(v):
                            if x not in block_pts:
                                best_x = x
                                break
                        
                    block_pts.append(best_x)
                
                add_block(block_pts)
            
            # Continue with standard greedy for remaining blocks
            while uncovered:
                # Pick lexicographically smallest uncovered t-subset (deterministic anchor)
                start_mask = min(uncovered)
                start_pts = mask_to_points(start_mask)
                
                block_pts = start_pts[:]
                
                while len(block_pts) < k:
                    best_gain = -1
                    candidates = []
                    
                    t_minus_1_subs = list(itertools.combinations(block_pts, t-1))
                    
                    for x in range(v):
                        if x in block_pts:
                            continue
                        
                        gain = 0
                        for sub in t_minus_1_subs:
                            m = 0
                            for p in sub:
                                m |= (1 << p)
                            m |= (1 << x)
                            if m in uncovered:
                                gain += 1
                        
                        if gain > best_gain:
                            best_gain = gain
                            candidates = [x]
                        elif gain == best_gain:
                            candidates.append(x)
                    
                    if candidates:
                        best_x = random.choice(candidates)
                    else:
                        for x in range(v):
                            if x not in block_pts:
                                best_x = x
                                break
                        
                    block_pts.append(best_x)
                
                add_block(block_pts)

        # Phase 2: Pruning
        prune_blocks()
        
        return blocks

    # Run multiple times with different seeds and keep the best
    best_blocks = None
    for s in range(10): # 10 restarts
        blocks = run_greedy(s)
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks
            if len(best_blocks) <= 300: # Early stop if we hit a very good bound? 
                # Known best is 335. 300 is very good.
                # But we should probably run all to be sure.
                pass

    return best_blocks
