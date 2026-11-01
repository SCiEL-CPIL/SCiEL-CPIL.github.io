import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tsubs_set = set(all_tsubs)
    
    def block_triples(block):
        return set(itertools.combinations(block, t))

    def random_block_containing(required):
        block = set(required)
        remaining = [x for x in range(v) if x not in block]
        random.shuffle(remaining)
        block.update(remaining[:k - len(block)])
        return sorted(block)

    def greedy_construct(seed):
        random.seed(seed)
        uncovered = set(all_tsubs_set)
        blocks = []
        while uncovered:
            start_triple = random.choice(list(uncovered))
            block = random_block_containing(start_triple)
            blocks.append(block)
            uncovered.difference_update(block_triples(block))
        return blocks

    def repair_covering(blocks, seed):
        random.seed(seed)
        covered = set()
        for b in blocks:
            covered.update(block_triples(b))
        uncovered = all_tsubs_set - covered
        
        while uncovered:
            start_triple = random.choice(list(uncovered))
            block = random_block_containing(start_triple)
            blocks.append(block)
            uncovered.difference_update(block_triples(block))
        return blocks

    def local_search(blocks):
        """
        Greedy deletion local search using coverage counts for speed.
        """
        coverage = {}
        for ts in all_tsubs_set:
            coverage[ts] = 0
        for b in blocks:
            for ts in block_triples(b):
                coverage[ts] += 1
        
        changed = True
        while changed:
            changed = False
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            for idx in indices:
                block = blocks[idx]
                essential = False
                for ts in block_triples(block):
                    if coverage[ts] == 1:
                        essential = True
                        break
                if not essential:
                    for ts in block_triples(block):
                        coverage[ts] -= 1
                    blocks.pop(idx)
                    changed = True
                    break # Restart loop
        return blocks

    # Initialize
    current_blocks = greedy_construct(42)
    current_blocks = local_search(current_blocks)
    
    best_blocks = list(current_blocks)
    best_count = len(current_blocks)
    
    temperature = 1.0
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit:
        # Perturbation
        perturbed = list(current_blocks)
        
        # Remove a random number of blocks
        num_remove = random.randint(1, 5)
        indices = random.sample(range(len(perturbed)), min(num_remove, len(perturbed)))
        for idx in sorted(indices, reverse=True):
            perturbed.pop(idx)
            
        # Repair
        perturbed = repair_covering(perturbed, random.randint(0, 10000))
        
        # Local Search
        perturbed = local_search(perturbed)
        
        # Acceptance
        delta = len(perturbed) - len(current_blocks)
        
        if delta < 0:
            # Improvement
            current_blocks = perturbed
            if len(current_blocks) < best_count:
                best_count = len(current_blocks)
                best_blocks = list(current_blocks)
        else:
            # Worse, accept with probability exp(-delta / temperature)
            if random.random() < pow(2.718, -delta / temperature):
                current_blocks = perturbed
                
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    return best_blocks
