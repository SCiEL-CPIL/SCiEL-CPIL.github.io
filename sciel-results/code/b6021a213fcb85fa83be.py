import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Safety margin

    # Precompute all t-subsets
    all_ts = list(itertools.combinations(range(v), t))
    # Map t-subset to index for faster array access? 
    # Set of tuples is fine for v=24.
    
    def greedy_construct(seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = set(all_ts)
        blocks = []
        
        while uncovered:
            # Pick a starting subset
            if seed is None:
                first_uncovered = min(uncovered)
            else:
                first_uncovered = random.choice(list(uncovered))
            
            block = list(first_uncovered)
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                # Optimization: Pre-compute combinations of block of size t-1
                # This changes as block grows, so we do it inside the loop
                block_combos = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in block_combos:
                        s = tuple(sorted(rest + (x,)))
                        if s in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        # If gain is max possible, we could break? 
                        # Max gain is len(block_combos)
                        if best_gain == len(block_combos):
                            break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            for s in itertools.combinations(block, t):
                if s in uncovered:
                    uncovered.remove(s)
                    
        return blocks

    def remove_redundant(blocks):
        cover_count = {}
        for block in blocks:
            for s in itertools.combinations(block, t):
                cover_count[s] = cover_count.get(s, 0) + 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                block = blocks[i]
                is_redundant = True
                for s in itertools.combinations(block, t):
                    if cover_count[s] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s in itertools.combinations(block, t):
                        cover_count[s] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        return blocks

    def pairwise_merge(blocks):
        # Returns True if a merge was performed
        if len(blocks) < 2:
            return False
        
        cover_count = {}
        for block in blocks:
            for s in itertools.combinations(block, t):
                cover_count[s] = cover_count.get(s, 0) + 1
        
        n = len(blocks)
        for i in range(n):
            if time.time() - start_time > time_limit:
                return False
            for j in range(i + 1, n):
                # Check if merging Bi and Bj is possible
                # Subsets that become uncovered if we remove Bi and Bj
                unique_to_pair = []
                for s in itertools.combinations(blocks[i], t):
                    if cover_count[s] <= 2: # Covered by Bi or Bj, maybe both
                        # Check if covered by anyone else
                        # cover_count[s] includes Bi and Bj.
                        # If cover_count[s] == 1, it's only in one of them (or both? no, count is total).
                        # If cover_count[s] == 2, it could be in both, or one in Bi and one in Bj, or one in Bi and one in other?
                        # We need to know if ANY OTHER block covers s.
                        # This requires more info than just count.
                        # Let's just check explicitly.
                        pass
                
                # Explicit check is safer but slower.
                # For v=24, n~80, pairs ~3000.
                # Checking coverage for each subset in Bi U Bj.
                # Size of Bi U Bj is at most 14. Subsets ~ 364.
                # 3000 * 364 * (check if in other blocks)
                # "Check if in other blocks" is O(n) if we scan all blocks.
                # 3000 * 364 * 80 = 87 million ops. Too slow in Python.
                
                # Optimization:
                # We can precompute for each subset s, the list of blocks covering it?
                # Or just the count.
                # If count[s] == 1, it's unique to one block. If that block is Bi or Bj, it's unique to the pair.
                # If count[s] == 2, it might be covered by Bi and Bj. If so, removing both makes it uncovered.
                # If count[s] == 2, and one is Bi, the other is not Bj, then removing Bi and Bj leaves it covered by the other.
                
                # Let's refine the condition:
                # A subset s is lost if:
                # 1. s is in Bi or Bj.
                # 2. No block other than Bi and Bj covers s.
                
                # We can check this by:
                # For s in combinations(Bi, t) U combinations(Bj, t):
                #    if cover_count[s] > 2: continue # Covered by >=3 blocks, so at least 1 other than Bi,Bj?
                #    # Wait, if cover_count is 3, it could be Bi, Bj, and Bk. Removing Bi,Bj leaves Bk. Safe.
                #    # If cover_count is 2, it could be Bi and Bj. Removing both leaves 0. Unsafe.
                #    # If cover_count is 2, it could be Bi and Bk. Removing Bi,Bj leaves Bk. Safe.
                #    # So we need to know if the 2 coverers are exactly Bi and Bj.
                
                # This is getting complicated. Let's stick to a simpler local search or just Greedy+Remove.
                # Given the time, Greedy+Remove is a solid improvement.
                pass
                
        return False

    # Main logic
    best_blocks = None
    best_len = float('inf')
    
    # 1. Deterministic Greedy + Remove
    blocks = greedy_construct()
    blocks = remove_redundant(blocks)
    best_blocks = blocks
    best_len = len(blocks)
    
    # 2. Random Restarts
    # Try a few random greedy constructions
    attempts = 0
    while time.time() - start_time < time_limit:
        seed = random.randint(0, 1000000)
        blocks = greedy_construct(seed)
        blocks = remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            # If we found a very good one, we might want to try to improve it further?
            # For now, just keep it.
            
        attempts += 1
        # If we've tried enough and time is running out, break
        if attempts > 1000: # Safety
            break

    return best_blocks
