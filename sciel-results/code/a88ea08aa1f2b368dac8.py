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

    def local_search(blocks):
        # Returns improved blocks
        blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            
            # 1. Remove redundant blocks
            cover_count = {}
            for block in blocks:
                for s in itertools.combinations(block, t):
                    cover_count[s] = cover_count.get(s, 0) + 1
            
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
                    break # Restart loop as indices changed
            
            if changed:
                continue

            # 2. Try to merge pairs of blocks
            # Recompute cover_count as it might have changed or if we didn't enter the above block
            if not cover_count: # If we didn't compute it in the first part? 
                cover_count = {}
                for block in blocks:
                    for s in itertools.combinations(block, t):
                        cover_count[s] = cover_count.get(s, 0) + 1

            # We need to know for each triple, which blocks cover it?
            # Or just count is enough for the "subset of {Bi, Bj}" check?
            # If count[t] == 1, and t is in Bi, it's covered only by Bi.
            # If count[t] == 2, and t is in Bi and Bj, it's covered by Bi and Bj (and no one else).
            # If count[t] == 2, and t is in Bi and Bk, it's covered by Bi and Bk. Removing Bi, Bj leaves Bk. Safe.
            
            # So, T_crit for pair (i, j) consists of triples s such that:
            # s is in Bi U Bj
            # AND (count[s] == 1 and s is in Bi) -> wait, if s in Bi and count 1, it's only in Bi.
            # AND (count[s] == 1 and s is in Bj)
            # AND (count[s] == 2 and s is in Bi and s is in Bj)
            
            # Actually, simpler:
            # A triple s is lost if C(s) \ {Bi, Bj} is empty.
            # This is equivalent to:
            # count[s] == 1 and s in {Bi, Bj} (specifically, s is in the one that covers it)
            # OR count[s] == 2 and s in Bi and s in Bj.
            
            # Let's find such triples.
            # If no such triples, we can remove one block (redundancy, already handled).
            # If there are such triples, can we form a new block B_new of size k containing all of them?
            
            n = len(blocks)
            merge_found = False
            for i in range(n):
                if time.time() - start_time > time_limit:
                    return blocks
                for j in range(i + 1, n):
                    # Collect critical triples
                    crit_triples = []
                    
                    # Check triples in Bi
                    for s in itertools.combinations(blocks[i], t):
                        if cover_count[s] == 1:
                            # Must be in Bi (since it's in Bi)
                            crit_triples.append(s)
                        elif cover_count[s] == 2:
                            # Check if also in Bj
                            if s in itertools.combinations(blocks[j], t):
                                crit_triples.append(s)
                    
                    # Check triples in Bj that are NOT in Bi (to avoid duplicates)
                    # Actually, the logic above covers:
                    # 1. s in Bi, count 1 -> Crit
                    # 2. s in Bi, count 2, s in Bj -> Crit
                    # What if s in Bj, count 1? It will be caught if we iterate Bj?
                    # My loop only iterates Bi. If s is in Bj but not Bi, and count 1, it is NOT in crit_triples from Bi loop.
                    # We need to check Bj as well.
                    
                    # Let's do a unified check.
                    # Union of triples in Bi and Bj.
                    # To avoid double counting and complexity, let's just check all triples in Bi U Bj.
                    
                    # Optimization:
                    # If len(crit_triples) > k, skip.
                    
                    # Let's re-implement the critical triple finding properly.
                    crit_set = set()
                    
                    for s in itertools.combinations(blocks[i], t):
                        if cover_count[s] == 1:
                            crit_set.add(s)
                        elif cover_count[s] == 2:
                            # Check if in blocks[j]
                            # s is a tuple of t sorted ints.
                            # Check if s is a subset of blocks[j]
                            if all(x in blocks[j] for x in s):
                                crit_set.add(s)
                    
                    for s in itertools.combinations(blocks[j], t):
                        if cover_count[s] == 1:
                            crit_set.add(s)
                        elif cover_count[s] == 2:
                            if all(x in blocks[i] for x in s):
                                crit_set.add(s)
                    
                    # Note: If s is in both Bi and Bj and count is 2, it's added twice, set handles it.
                    # If s is in Bi, count 1, added.
                    # If s is in Bj, count 1, added.
                    
                    if not crit_set:
                        # Should have been caught by redundancy removal?
                        # If Bi and Bj are not redundant individually, but together they are?
                        # No, if no triple is critical, it means every triple in Bi U Bj is covered by >= 1 other block.
                        # Then Bi is redundant? No.
                        # If every triple in Bi is covered by other blocks (not Bj), Bi is redundant.
                        # If every triple in Bi is covered by other blocks (including Bj), Bi is redundant?
                        # Yes, if C(s) \ {Bi} is not empty for all s in Bi, Bi is redundant.
                        # So if crit_set is empty, Bi or Bj should be redundant.
                        # Since redundancy removal runs first, this shouldn't happen unless my logic is flawed.
                        # Wait, redundancy removal checks if *any* triple in Bi has count 1.
                        # If count is 2 for all, and covered by others, it's redundant.
                        # So crit_set empty implies redundancy.
                        continue

                    if len(crit_set) > k:
                        continue

                    # Try to construct a new block B_new containing all crit_set elements
                    # We need to pick k - len(crit_set) additional points.
                    # To maximize future mergeability or coverage, we might want to pick points that are "common".
                    # But for now, just pick valid points.
                    # Points in crit_set are distinct?
                    # crit_set contains t-subsets. The union of points in crit_set might be > k?
                    # No, each crit triple has t points.
                    # We need a block of size k containing ALL these triples.
                    # This means the union of points in all crit_triples must be a subset of B_new.
                    # So, let P = union of all points in crit_set.
                    # If len(P) > k, impossible.
                    
                    P = set()
                    for s in crit_set:
                        P.update(s)
                    
                    if len(P) > k:
                        continue

                    # We need to add k - len(P) points from V \ P.
                    # Any choice works to form a valid block.
                    # Let's pick the smallest available points to be deterministic?
                    # Or random?
                    # Random might lead to different structures.
                    # Let's pick random points to avoid bias.
                    available = [x for x in range(v) if x not in P]
                    if len(available) < k - len(P):
                        continue
                    
                    # Pick k - len(P) points
                    extra = random.sample(available, k - len(P))
                    B_new = sorted(list(P) + extra)
                    
                    # Replace Bi and Bj with B_new
                    # Remove i and j (higher index first)
                    blocks.pop(j)
                    blocks.pop(i)
                    blocks.append(B_new)
                    
                    changed = True
                    merge_found = True
                    break # Restart local search loop
            
            if merge_found:
                continue

        return blocks

    # Main logic
    best_blocks = None
    best_len = float('inf')
    
    # 1. Deterministic Greedy + Local Search
    blocks = greedy_construct()
    blocks = local_search(blocks)
    best_blocks = blocks
    best_len = len(blocks)
    
    # 2. Random Restarts
    # Try a few random greedy constructions
    attempts = 0
    while time.time() - start_time < time_limit:
        seed = random.randint(0, 1000000)
        blocks = greedy_construct(seed)
        blocks = local_search(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        attempts += 1
        if attempts > 1000: # Safety
            break

    return best_blocks
