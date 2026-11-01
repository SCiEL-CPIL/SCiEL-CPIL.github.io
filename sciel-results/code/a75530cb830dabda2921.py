import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115
    
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tsubs_set = set(all_tsubs)
    
    def block_triples(block):
        return set(itertools.combinations(block, t))

    def greedy_construct(seed_val):
        random.seed(seed_val)
        uncovered = set(all_tsubs)
        blocks = []
        while uncovered:
            start_triple = random.choice(list(uncovered))
            block = list(start_triple)
            random.shuffle(block)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                candidates = [x for x in range(v) if x not in block]
                random.shuffle(candidates)
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(block_triples(block))
        return blocks

    def local_search(blocks):
        coverage = {ts: 0 for ts in all_tsubs}
        for b in blocks:
            for ts in block_triples(b):
                coverage[ts] += 1
        
        improved = True
        while improved:
            improved = False
            # Deletion
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            for idx in indices:
                b = blocks[idx]
                ok = True
                for ts in block_triples(b):
                    if coverage[ts] <= 1:
                        ok = False
                        break
                if ok:
                    for ts in block_triples(b):
                        coverage[ts] -= 1
                    blocks.pop(idx)
                    improved = True
                    break
            
            if not improved:
                # 1-Swap
                for idx in range(len(blocks)):
                    b = blocks[idx]
                    for i, old_x in enumerate(b):
                        for new_x in range(v):
                            if new_x == old_x or new_x in b:
                                continue
                            old_triples = block_triples(b)
                            new_b = sorted([new_x] + [p for p in b if p != old_x])
                            new_triples = block_triples(new_b)
                            
                            valid = True
                            for ts in old_triples:
                                if coverage[ts] <= 1 and ts not in new_triples:
                                    valid = False
                                    break
                            if valid:
                                for ts in old_triples:
                                    coverage[ts] -= 1
                                for ts in new_triples:
                                    coverage[ts] += 1
                                blocks[idx] = new_b
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        return blocks

    # ILS
    best_blocks = greedy_construct(42)
    best_blocks = local_search(best_blocks)
    best_len = len(best_blocks)
    
    current_blocks = list(best_blocks)
    
    while time.time() - start_time < time_limit:
        # Perturb
        perturb_count = random.randint(1, 3)
        for _ in range(perturb_count):
            idx = random.randint(0, len(current_blocks) - 1)
            # Replace block with a random one
            new_block = sorted(random.sample(range(v), k))
            current_blocks[idx] = new_block
        
        # Local Search
        new_blocks = local_search(current_blocks)
        
        # Accept
        if len(new_blocks) < best_len:
            best_blocks = new_blocks
            best_len = len(new_blocks)
            current_blocks = list(new_blocks)
        elif len(new_blocks) == best_len:
            # Accept with some probability to diversify
            if random.random() < 0.5:
                current_blocks = list(new_blocks)
        else:
            # Reject, but keep current for next perturbation? 
            # Or revert to best?
            # Standard ILS: if worse, revert to pre-perturbation state?
            # Here we mutated current_blocks in place. 
            # Let's keep current_blocks as the result of LS on the perturbed best.
            pass

    return best_blocks
