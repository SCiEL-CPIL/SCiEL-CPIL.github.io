"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 140.0
    rng = random.Random(42)
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Coverage array: which blocks cover which t-subsets
    # We'll use a bitset-like approach with numpy
    
    def count_coverage(blocks):
        """Return array of coverage counts for each t-subset."""
        cov = np.zeros(n_tsubs, dtype=np.int32)
        for block in blocks:
            block_set = set(block)
            for ts in itertools.combinations(block, t):
                cov[tsub_to_idx[ts]] += 1
        return cov
    
    def is_covered(blocks):
        """Check if all t-subsets are covered."""
        covered = np.zeros(n_tsubs, dtype=bool)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                covered[tsub_to_idx[ts]] = True
        return covered.all()
    
    def uncovered_count(blocks):
        """Return number of uncovered t-subsets."""
        cov = count_coverage(blocks)
        return int(np.sum(cov == 0))
    
    # Phase 1: Generate candidate blocks using structured + random methods
    candidate_blocks = set()
    
    # Method A: Greedy construction
    def greedy_construct():
        uncovered_set = set(all_tsubs)
        blocks = []
        while uncovered_set:
            # Pick the t-subset with fewest covering options (hardest)
            # For simplicity, pick the lexicographically smallest
            target = min(uncovered_set)
            block = list(target)
            while len(block) < k:
                # Find point that covers most uncovered t-subsets
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            block_tuple = tuple(block)
            blocks.append(block)
            # Remove covered t-subsets
            for ts in itertools.combinations(block, t):
                uncovered_set.discard(ts)
        return blocks
    
    # Method B: Random blocks biased toward covering uncovered sets
    def random_construct(iterations=50):
        blocks = []
        uncovered_set = set(all_tsubs)
        for _ in range(iterations):
            if not uncovered_set:
                break
            # Pick a random uncovered t-subset as seed
            target = rng.choice(list(uncovered_set)[:1000]) if len(uncovered_set) > 1000 else rng.choice(list(uncovered_set))
            block = list(target)
            # Fill remaining with random points
            remaining = [x for x in range(v) if x not in block]
            rng.shuffle(remaining)
            for x in remaining:
                if len(block) >= k:
                    break
                block.append(x)
            block.sort()
            block_tuple = tuple(block)
            if block_tuple not in set(map(tuple, blocks)):
                blocks.append(block)
                for ts in itertools.combinations(block, t):
                    uncovered_set.discard(ts)
        return blocks
    
    # Phase 1: Build initial covering with greedy
    initial_blocks = greedy_construct()
    current_blocks = list(set(map(tuple, initial_blocks)))
    
    if time.time() > deadline - 30:
        return [list(b) for b in current_blocks]
    
    # Phase 2: Local search - try to remove redundant blocks
    def remove_redundant(blocks):
        blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                # Check if block i is redundant
                without_i = [b for j, b in enumerate(blocks) if j != i]
                if is_covered(without_i):
                    blocks.pop(i)
                    changed = True
                    break
        return blocks
    
    current_blocks = remove_redundant(current_blocks)
    
    if time.time() > deadline - 20:
        return [list(b) for b in current_blocks]
    
    # Phase 3: Local search - swap blocks to improve
    best_blocks = list(current_blocks)
    best_size = len(best_blocks)
    
    # Generate additional candidate blocks
    extra_candidates = []
    for _ in range(200):
        if time.time() > deadline - 15:
            break
        # Random block
        block = sorted(rng.sample(range(v), k))
        extra_candidates.append(tuple(block))
    
    # Try adding and removing
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        
        # Try removing a random block
        if len(current_blocks) > 10:
            idx = rng.randrange(len(current_blocks))
            trial = [b for j, b in enumerate(current_blocks) if j != idx]
            if is_covered(trial):
                current_blocks = trial
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
        
        # Try swapping a block
        if len(current_blocks) > 10 and extra_candidates:
            idx = rng.randrange(len(current_blocks))
            new_block = rng.choice(extra_candidates)
            if new_block not in current_blocks:
                trial = list(current_blocks)
                trial[idx] = new_block
                if is_covered(trial):
                    current_blocks = trial
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
    
    # Final redundancy removal
    best_blocks = remove_redundant(best_blocks)
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
