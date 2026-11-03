import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for 120s limit
    
    # Precompute all t-subsets as integer IDs for fast hashing
    # ID = p0 * v^2 + p1 * v + p2 for t=3
    # General ID encoding for t-subsets
    def encode_subset(sub):
        code = 0
        for p in sub:
            code = code * v + p
        return code

    all_subs = [tuple(sorted(c)) for c in itertools.combinations(range(v), t)]
    sub_ids = {encode_subset(s): s for s in all_subs}
    # Map from tuple to id for fast lookup
    tuple_to_id = {s: i for i, s in enumerate(all_subs)}
    
    # Precompute triples for a given block (list of k points)
    def get_block_triples(block):
        return [tuple(sorted(c)) for c in itertools.combinations(block, t)]

    def greedy_covering(seed):
        random.seed(seed)
        uncovered = set(range(len(all_subs)))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple to start the block
            # Using random.choice for diversity
            start_triple_id = random.choice(list(uncovered))
            start_triple = all_subs[start_triple_id]
            
            block = list(start_triple)
            
            while len(block) < k:
                # Find point that covers most uncovered triples
                # Candidates are points not in block
                candidates = [x for x in range(v) if x not in block]
                
                # Calculate gain for each candidate
                # Gain = number of uncovered triples formed by (t-1) subset of block + candidate
                # For t=3, we form triples from pairs in block + candidate
                best_gain = -1
                best_candidates = []
                
                # Optimization: Precompute pairs in block
                block_pairs = list(itertools.combinations(block, t-1))
                
                for x in candidates:
                    gain = 0
                    for pair in block_pairs:
                        # Form triple
                        tri = tuple(sorted(pair + (x,)))
                        if tri in tuple_to_id:
                            tid = tuple_to_id[tri]
                            if tid in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain and gain > 0:
                        best_candidates.append(x)
                
                if not best_candidates:
                    # Should not happen if uncovered is not empty, but safety
                    x = random.choice(candidates)
                else:
                    # Pick randomly from top candidates to add diversity
                    x = random.choice(best_candidates)
                
                block.append(x)
            
            block.sort()
            blocks.append(block)
            
            # Remove covered triples from uncovered
            for tri in get_block_triples(block):
                tid = tuple_to_id[tri]
                uncovered.discard(tid)
                
        return blocks

    def prune_covering(blocks):
        # Maintain coverage counts
        # size = len(all_subs)
        counts = [0] * len(all_subs)
        
        # Initial counts
        for b in blocks:
            for tri in get_block_triples(b):
                tid = tuple_to_id[tri]
                counts[tid] += 1
        
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to allow removal during iteration (or copy)
            # We iterate through indices
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                # Check if block is essential
                # Decrement counts
                is_essential = False
                affected_tids = []
                for tri in get_block_triples(b):
                    tid = tuple_to_id[tri]
                    counts[tid] -= 1
                    affected_tids.append(tid)
                    if counts[tid] == 0:
                        is_essential = True
                        break
                
                if is_essential:
                    # Restore counts
                    for tid in affected_tids:
                        counts[tid] += 1
                else:
                    # Keep removed
                    blocks.pop(i)
                    changed = True
        
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    while time.time() < deadline:
        seed += 1
        raw_blocks = greedy_covering(seed)
        pruned_blocks = prune_covering(raw_blocks)
        
        if len(pruned_blocks) < best_len:
            best_len = len(pruned_blocks)
            best_blocks = pruned_blocks
            
        # If we hit a very good solution, we might spend more time trying to improve it?
        # For now, just keep running new random seeds.
        # 76 is the target. If we hit 76, we can try to go lower, but 69 is the bound.
        
    if best_blocks is None:
        # Fallback: run one more time if time allows, or return empty (invalid)
        # But greedy should always produce a valid cover.
        raw_blocks = greedy_covering(0)
        best_blocks = prune_covering(raw_blocks)
        
    return best_blocks
