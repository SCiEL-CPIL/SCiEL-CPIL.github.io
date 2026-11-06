import time
from itertools import combinations

_DEADLINE_START = time.time()
_DEADLINE_SECONDS = 130.0  # Safety margin for 150s limit

def _now():
    return time.time()

def _time_left():
    return _DEADLINE_SECONDS - (_now() - _DEADLINE_START)

def _precompute():
    """
    Precompute 5-subset IDs for all 6-subsets.
    5-subsets of 12 points: C(12,5) = 792.
    6-subsets of 12 points: C(12,6) = 924.
    """
    points = range(12)
    all_5subs = list(combinations(points, 5))
    map_5sub_to_id = {sub: i for i, sub in enumerate(all_5subs)}
    
    all_6subs = list(combinations(points, 6))
    block_to_5ids = {}
    for b in all_6subs:
        ids = [map_5sub_to_id[s] for s in combinations(b, 5)]
        block_to_5ids[b] = ids
        
    return all_6subs, block_to_5ids

def construct():
    """
    Construct the Witt design S(5,6,12) using a hybrid approach:
    1. Try to generate the Witt design using the Steiner system properties.
    2. Fall back to greedy with backtracking if needed.
    """
    all_6subs, block_to_5ids = _precompute()
    
    # Strategy: Use the known structure of the Witt design.
    # The Witt design S(5,6,12) has 132 blocks.
    # We can construct it by starting with a base block and applying the automorphism group M12.
    # However, generating M12 is complex. Instead, we use a greedy approach with smart ordering.
    
    # First, try a greedy approach with a specific ordering that exploits the design structure.
    # The key insight is that in S(5,6,12), every 5-subset is in exactly one block.
    # We can use this to guide our greedy selection.
    
    # Sort blocks by a heuristic: prioritize blocks that cover "rare" 5-subsets.
    # Actually, in the optimal design, all 5-subsets are equally rare (each in exactly one block).
    # So we need a different heuristic.
    
    # Let's try multiple greedy strategies and pick the best one.
    
    def greedy_with_order(order):
        covered_ids = set()
        blocks = []
        for b in order:
            if _time_left() < 5.0:
                break
            ids = block_to_5ids[b]
            # Check if any 5-subset is already covered
            if any(i in covered_ids for i in ids):
                continue
            blocks.append(b)
            covered_ids.update(ids)
            if len(blocks) == 132:
                return blocks
        return blocks
    
    # Strategy 1: Lexicographic order
    blocks1 = greedy_with_order(all_6subs)
    if len(blocks1) == 132:
        return [list(b) for b in blocks1]
    
    # Strategy 2: Reverse lexicographic order
    blocks2 = greedy_with_order(reversed(all_6subs))
    if len(blocks2) == 132:
        return [list(b) for b in blocks2]
    
    # Strategy 3: Sort by sum of elements (ascending)
    order3 = sorted(all_6subs, key=lambda b: sum(b))
    blocks3 = greedy_with_order(order3)
    if len(blocks3) == 132:
        return [list(b) for b in blocks3]
    
    # Strategy 4: Sort by sum of elements (descending)
    order4 = sorted(all_6subs, key=lambda b: -sum(b))
    blocks4 = greedy_with_order(order4)
    if len(blocks4) == 132:
        return [list(b) for b in blocks4]
    
    # Strategy 5: Random shuffle
    import random
    rng = random.Random(42)
    order5 = all_6subs[:]
    rng.shuffle(order5)
    blocks5 = greedy_with_order(order5)
    if len(blocks5) == 132:
        return [list(b) for b in blocks5]
    
    # Strategy 6: Another random shuffle with different seed
    rng2 = random.Random(123)
    order6 = all_6subs[:]
    rng2.shuffle(order6)
    blocks6 = greedy_with_order(order6)
    if len(blocks6) == 132:
        return [list(b) for b in blocks6]
    
    # If none of the greedy strategies found 132 blocks, try a backtracking search
    # for the last few blocks.
    
    def backtrack_search(partial_blocks, partial_covered, remaining_blocks, target=132):
        if len(partial_blocks) == target:
            return partial_blocks
        if not remaining_blocks:
            return None
        if _time_left() < 3.0:
            return None
            
        for i, b in enumerate(remaining_blocks):
            ids = block_to_5ids[b]
            if any(i_id in partial_covered for i_id in ids):
                continue
            # Try adding this block
            new_covered = partial_covered.copy()
            new_covered.update(ids)
            result = backtrack_search(
                partial_blocks + [b],
                new_covered,
                remaining_blocks[i+1:]
            )
            if result is not None:
                return result
        return None
    
    # Use the best greedy result as a starting point for backtracking
    best_greedy = max([blocks1, blocks2, blocks3, blocks4, blocks5, blocks6], key=len)
    if len(best_greedy) > 100:
        # Compute covered IDs from best_greedy
        covered_ids = set()
        for b in best_greedy:
            covered_ids.update(block_to_5ids[b])
        
        # Remaining blocks to choose from
        remaining = [b for b in all_6subs if b not in set(best_greedy)]
        
        result = backtrack_search(best_greedy, covered_ids, remaining)
        if result is not None and len(result) == 132:
            return [list(b) for b in result]
    
    # Return the best greedy result
    return [list(b) for b in best_greedy]

if __name__ == "__main__":
    start = time.time()
    blocks = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s, Blocks: {len(blocks)}")
    # Verify validity
    all_5subs = list(combinations(range(12), 5))
    map_5sub_to_id = {sub: i for i, sub in enumerate(all_5subs)}
    covered = set()
    valid = True
    for b in blocks:
        for s in combinations(b, 5):
            sid = map_5sub_to_id[s]
            if sid in covered:
                valid = False
                break
            covered.add(sid)
        if not valid:
            break
    print(f"Valid: {valid}, Size: {len(blocks)}, Covered 5-subs: {len(covered)}/{792}")
