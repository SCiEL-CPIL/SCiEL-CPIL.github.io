    def greedy_cover_partial(initial_blocks, initial_counts, uncovered_set, seed):
        random.seed(seed)
        blocks = list(initial_blocks)
        counts = list(initial_counts)
        uncovered = set(uncovered_set)
        
        while uncovered:
            target_triple_idx = random.choice(list(uncovered))
            target_triple = all_triples[target_triple_idx]
            
            best_gain = -1
            best_block = None
            
            target_set = set(target_triple)
            remaining_points = [p for p in points if p not in target_set]
            
            for combo in itertools.combinations(remaining_points, k - t):
                candidate_block = tuple(sorted(target_triple + combo))
                gain = 0
                for tr_idx in get_block_triples(candidate_block):
                    if tr_idx in uncovered:
                        gain += 1
                        if gain == 35: break
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
                    if best_gain == 35: break
            
            if best_block:
                blocks.append(best_block)
                for tr_idx in get_block_triples(best_block):
                    counts[tr_idx] += 1
                    if counts[tr_idx] == 1:
                        uncovered.discard(tr_idx)
                        
        return blocks, counts
