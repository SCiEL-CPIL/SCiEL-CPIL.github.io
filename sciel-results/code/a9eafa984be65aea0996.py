import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s kill
    
    # For the specific case v=24, k=7, t=3, use structured approach
    # General fallback: greedy
    if v != 24 or k != 7 or t != 3:
        # Fallback: simple greedy
        all_triples = list(itertools.combinations(range(v), t))
        triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
        all_blocks = list(itertools.combinations(range(v), k))
        uncovered = set(range(len(all_triples)))
        selected = []
        while uncovered:
            if time.time() > deadline - 5:
                break
            best_block = None
            best_count = -1
            for b in all_blocks:
                if b in selected:
                    continue
                count = 0
                for tri in itertools.combinations(b, t):
                    if triple_to_idx[tri] in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = b
            if best_block is None or best_count == 0:
                break
            selected.append(best_block)
            for tri in itertools.combinations(best_block, t):
                uncovered.discard(triple_to_idx[tri])
        return selected
    
    # Structured approach for C(24,7,3)
    # Divide 24 points into 4 groups of 6
    # Points: 0-5 (G0), 6-11 (G1), 12-17 (G2), 18-23 (G3)
    groups = [list(range(6)), list(range(6,12)), list(range(12,18)), list(range(18,24))]
    
    # Step 1: Construct blocks that cover all "mixed" triples (triples not all in one group)
    # A triple is mixed if its points come from >= 2 different groups.
    # We need blocks of size 7 that cover these.
    # Strategy: Use a systematic construction.
    # Each block will have points from multiple groups.
    # Let's use a round-robin / cyclic construction.
    
    # Actually, let's think about this differently.
    # Total triples: C(24,3) = 2024
    # Internal triples per group: C(6,3) = 20, so 4*20 = 80 internal triples
    # Mixed triples: 2024 - 80 = 1944
    
    # Each block of size 7 covers C(7,3) = 35 triples.
    # If we use N blocks, they cover at most 35*N triples total (with overlaps).
    # Lower bound: ceil(2024/35) = 58. But we need 76 known.
    
    # Let's use a different structured approach:
    # 1. Cover all mixed triples with a GDD-like structure
    # 2. Cover internal triples with additional blocks
    
    # For mixed triples: A triple is mixed if it has points from at least 2 groups.
    # We can construct blocks where each block has at most 2 points from any one group,
    # or use a more flexible approach.
    
    # Let me try a direct constructive approach:
    # Generate candidate blocks systematically, then use greedy to select.
    
    # Precompute all triples and their group signatures
    all_triples = list(itertools.combinations(range(24), 3))
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Determine which triples are internal (all in same group)
    def get_group(p):
        return p // 6
    
    internal_triples = set()
    for tri in all_triples:
        if len(set(get_group(p) for p in tri)) == 1:
            internal_triples.add(triple_to_idx[tri])
    
    # Generate all possible blocks
    all_blocks = list(itertools.combinations(range(24), 7))
    num_blocks = len(all_blocks)
    
    # For each block, compute which triples it covers
    block_triples = []
    for b in all_blocks:
        tris = set()
        for tri in itertools.combinations(b, 3):
            tris.add(triple_to_idx[tri])
        block_triples.append(tris)
    
    # Phase 1: Greedy to cover all mixed triples
    mixed_triples = set(range(num_triples)) - internal_triples
    uncovered_mixed = set(mixed_triples)
    selected_blocks = []
    selected_set = set()
    
    # Precompute for speed: for each block, list of mixed triples it covers
    block_mixed_triples = []
    for i, b in enumerate(all_blocks):
        mixed_covered = set()
        for tri in itertools.combinations(b, 3):
            idx = triple_to_idx[tri]
            if idx in mixed_triples:
                mixed_covered.add(idx)
        block_mixed_triples.append(mixed_covered)
    
    while uncovered_mixed:
        if time.time() > deadline - 60:
            break
        best_gain = -1
        best_idx = -1
        for i in range(num_blocks):
            if i in selected_set:
                continue
            gain = len(block_mixed_triples[i] & uncovered_mixed)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1 or best_gain == 0:
            break
        selected_blocks.append(best_idx)
        selected_set.add(best_idx)
        uncovered_mixed -= block_mixed_triples[best_idx]
    
    # Phase 2: Cover remaining internal triples
    # Check which internal triples are still uncovered
    all_covered = set()
    for idx in selected_blocks:
        all_covered |= block_triples[idx]
    
    uncovered_internal = internal_triples - all_covered
    
    # Add blocks to cover uncovered internal triples
    while uncovered_internal:
        if time.time() > deadline - 20:
            break
        best_gain = -1
        best_idx = -1
        for i in range(num_blocks):
            if i in selected_set:
                continue
            gain = len(block_triples[i] & uncovered_internal)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1 or best_gain == 0:
            break
        selected_blocks.append(best_idx)
        selected_set.add(best_idx)
        uncovered_internal -= (block_triples[best_idx] & uncovered_internal)
    
    # Phase 3: Remove redundant blocks
    # Count how many times each triple is covered
    counts = [0] * num_triples
    for idx in selected_blocks:
        for ti in block_triples[idx]:
            counts[ti] += 1
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        to_remove = []
        for idx in selected_blocks:
            redundant = True
            for ti in block_triples[idx]:
                if counts[ti] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(idx)
        
        if to_remove:
            for idx in to_remove:
                selected_blocks.remove(idx)
                selected_set.remove(idx)
                for ti in block_triples[idx]:
                    counts[ti] -= 1
            improved = True
    
    # Phase 4: Try 1-swap improvements
    if time.time() < deadline - 3:
        # For each block, try to find a replacement that allows removing another block
        # Simplified: try removing each block and see if coverage is maintained
        # by checking if all its unique triples are covered by others
        for _ in range(3):  # Multiple passes
            if time.time() > deadline - 3:
                break
            changed = False
            for i in range(len(selected_blocks)):
                if time.time() > deadline - 3:
                    break
                idx = selected_blocks[i]
                # Check if removing this block leaves all triples covered
                can_remove = True
                for ti in block_triples[idx]:
                    if counts[ti] == 1:
                        can_remove = False
                        break
                if can_remove:
                    selected_blocks.pop(i)
                    selected_set.remove(idx)
                    for ti in block_triples[idx]:
                        counts[ti] -= 1
                    changed = True
                    break  # Restart after removal
    
    final_blocks = [all_blocks[idx] for idx in selected_blocks]
    
    # Final validation
    covered = set()
    for b in final_blocks:
        for tri in itertools.combinations(b, 3):
            covered.add(tri)
    
    # If not fully covered (shouldn't happen, but safety), fall back
    if len(covered) < num_triples:
        # Add blocks until covered
        remaining = set(all_triples) - covered
        for b in all_blocks:
            if time.time() > deadline - 1:
                break
            if b in final_blocks:
                continue
            if any(tri in remaining for tri in itertools.combinations(b, 3)):
                final_blocks.append(b)
                for tri in itertools.combinations(b, 3):
                    covered.add(tri)
                remaining -= set(itertools.combinations(b, 3))
                if not remaining:
                    break
    
    return final_blocks
