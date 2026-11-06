import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)
    
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) configurations
    configs = []
    config_id_map = {}
    triples_in_config = []
    
    points = range(N)
    for pts in itertools.combinations(points, 6):
        p0, p1, p2, p3, p4, p5 = pts
        pairs = [
            [(p0, p1), (p2, p3), (p4, p5)],
            [(p0, p1), (p2, p4), (p3, p5)],
            [(p0, p1), (p2, p5), (p3, p4)],
            [(p0, p2), (p1, p3), (p4, p5)],
            [(p0, p2), (p1, p4), (p3, p5)],
            [(p0, p2), (p1, p5), (p3, p4)],
            [(p0, p3), (p1, p2), (p4, p5)],
            [(p0, p3), (p1, p4), (p2, p5)],
            [(p0, p3), (p1, p5), (p2, p4)],
            [(p0, p4), (p1, p2), (p3, p5)],
            [(p0, 4), (p1, p3), (p2, p5)], # Typo in original seed? No, p0, p4
            [(p0, p4), (p1, p5), (p2, p3)],
            [(p0, p5), (p1, p2), (p3, p4)],
            [(p0, p5), (p1, p3), (p2, p4)],
            [(p0, p5), (p1, p4), (p2, p3)]
        ]
        # Correcting the potential typo in my thought process, sticking to the logic:
        # The list of 15 pairings of 6 points into 3 pairs is standard.
        # Let's just generate them properly to be safe and fast.
        pass
        
    # Efficient generation of K(2,2,2) configs
    # A K(2,2,2) is defined by 3 disjoint pairs (a,b), (c,d), (e,f).
    # The 8 transversal triples are (a/c/e, a/c/f, a/d/e, a/d/f, b/c/e, b/c/f, b/d/e, b/d/f).
    # We need to ensure we don't duplicate configs.
    
    all_pairs = list(itertools.combinations(range(N), 2))
    pair_to_idx = {p: i for i, p in enumerate(all_pairs)}
    
    # Map triple to the set of pairs it "completes" or is part of?
    # Actually, it's easier to map Triple -> Configs as before, but we need the config list first.
    
    # Generate all unique sets of 3 disjoint pairs
    # C(16,2) = 120 pairs.
    # This might be slow if done naively.
    # The previous seed code did it by iterating 6-point subsets. C(16,6) = 8008.
    # Each 6-subset has 15 pairings. 8008 * 15 = 120120.
    # This is very fast.
    
    configs = []
    config_id_map = {}
    triples_in_config = []
    
    for pts in itertools.combinations(points, 6):
        p0, p1, p2, p3, p4, p5 = pts
        # 15 ways to partition 6 items into 3 pairs
        # (0,1)(2,3)(4,5)
        # (0,1)(2,4)(3,5)
        # (0,1)(2,5)(3,4)
        # (0,2)(1,3)(4,5)
        # (0,2)(1,4)(3,5)
        # (0,2)(1,5)(3,4)
        # (0,3)(1,2)(4,5)
        # (0,3)(1,4)(2,5)
        # (0,3)(1,5)(2,4)
        # (0,4)(1,2)(3,5)
        # (0,4)(1,3)(2,5)
        # (0,4)(1,5)(2,3)
        # (0,5)(1,2)(3,4)
        # (0,5)(1,3)(2,4)
        # (0,5)(1,4)(2,3)
        
        pairings = [
            ((p0, p1), (p2, p3), (p4, p5)),
            ((p0, p1), (p2, p4), (p3, p5)),
            ((p0, p1), (p2, p5), (p3, p4)),
            ((p0, p2), (p1, p3), (p4, p5)),
            ((p0, p2), (p1, p4), (p3, p5)),
            ((p0, p2), (p1, p5), (p3, p4)),
            ((p0, p3), (p1, p2), (p4, p5)),
            ((p0, p3), (p1, p4), (p2, p5)),
            ((p0, p3), (p1, p5), (p2, p4)),
            ((p0, p4), (p1, p2), (p3, p5)),
            ((p0, p4), (p1, p3), (p2, p5)),
            ((p0, p4), (p1, p5), (p2, p3)),
            ((p0, p5), (p1, p2), (p3, p4)),
            ((p0, p5), (p1, p3), (p2, p4)),
            ((p0, p5), (p1, p4), (p2, p3))
        ]
        
        for pair_set in pairings:
            norm_pairs = tuple(sorted(tuple(sorted(p)) for p in pair_set))
            if norm_pairs not in config_id_map:
                cid = len(configs)
                config_id_map[norm_pairs] = cid
                configs.append(norm_pairs)
                
                (a,b), (c,d), (e,f) = norm_pairs
                tris = []
                for x in (a,b):
                    for y in (c,d):
                        for z in (e,f):
                            t = tuple(sorted((x,y,z)))
                            tris.append(triple_to_idx[t])
                triples_in_config.append(tris)
                
    num_configs = len(configs)
    
    # Build reverse index: Triple -> List of Config IDs
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)
            
    # Data structures for the Greedy Set Packing
    present = np.zeros(num_triples, dtype=bool)
    config_counts = np.zeros(num_configs, dtype=np.int32)
    
    # For efficiency, we maintain a list of "available" triples
    # and for each triple, we calculate its "value" (number of new configs it would block)
    # Value(t) = sum(1 for cid in triple_to_configs[t] if config_counts[cid] < 7)
    # If we add t, it blocks any config where count was < 7 and becomes 8? No.
    # If we add t, count[cid] increases by 1.
    # If count[cid] was 7, it becomes 8 -> Config is VIOLATED.
    # We must NOT violate any config.
    # So a triple t is "valid" if for all cid in triple_to_configs[t], config_counts[cid] < 7.
    
    # The "Marginal Gain" heuristic:
    # We want to add triples. Adding a triple t consumes capacity in its configs.
    # To maximize the total number of triples, we should prioritize triples that
    # "waste" the least capacity or block the fewest other triples?
    # Actually, standard greedy for Max Independent Set / Set Packing:
    # Pick the element with the smallest "degree" (number of conflicts) first?
    # Or largest?
    # In Set Packing, we usually pick the set that conflicts with the fewest other sets (lowest degree) to leave more room.
    # Here, "conflict" is sharing a config.
    # Degree(t) = number of other triples sharing a config with t.
    # However, the capacity is 7 per config.
    
    # Let's try a different heuristic:
    # Greedy:
    # 1. Start with empty set.
    # 2. While there are valid triples:
    #    a. Find the triple with the highest "Score".
    #    b. Add it.
    #    c. Update counts.
    #    d. Remove invalid triples from consideration.
    
    # What is the best Score?
    # If we pick a triple that belongs to configs with high remaining capacity, it's "safe".
    # If we pick a triple that belongs to configs with low remaining capacity (e.g. count=6), it's "risky" (might block many others).
    # So we prefer triples that are in "loose" configs.
    # Score(t) = sum( (7 - config_counts[cid]) for cid in triple_to_configs[t] )
    # Higher score = more slack = better choice.
    
    # Let's implement this.
    
    # To make it faster, we can use a priority queue (heapq).
    # But the scores change as we add triples.
    # When we add triple t, it decreases the score of all triples sharing a config with t.
    # This is an update-heavy process.
    
    # Alternative: Randomized Greedy.
    # 1. Shuffle all triples.
    # 2. Iterate through them.
    # 3. If valid, add it.
    # This is O(N * Avg_Degree).
    # Let's try multiple random restarts with different shuffles, and also a "Weighted" greedy.
    
    def run_greedy(strategy="random", seed=0):
        """
        strategy: 'random' (shuffle order), 'degree' (lowest degree first), 'slack' (highest slack first)
        """
        present[:] = False
        config_counts[:] = 0
        current_set = set()
        
        if strategy == "random":
            rng = random.Random(seed)
            order = list(range(num_triples))
            rng.shuffle(order)
            
            for t_idx in order:
                if time.time() > deadline - 2:
                    break
                # Check validity
                valid = True
                for cid in triple_to_configs[t_idx]:
                    if config_counts[cid] >= 7:
                        valid = False
                        break
                if valid:
                    present[t_idx] = True
                    current_set.add(t_idx)
                    for cid in triple_to_configs[t_idx]:
                        config_counts[cid] += 1
                        
        elif strategy == "slack":
            # We need to dynamically select the best triple.
            # This is O(N^2) if done naively.
            # N=560. N^2 = 313,600.
            # Each check is O(Degree). Degree ~ 10-20.
            # Total ops ~ 6M. Very fast.
            # We can repeat this until no more valid triples.
            
            remaining = set(range(num_triples))
            
            while remaining:
                if time.time() > deadline - 2:
                    break
                
                best_t = -1
                best_score = -1
                
                # Find triple with max slack
                for t_idx in remaining:
                    score = 0
                    valid = True
                    for cid in triple_to_configs[t_idx]:
                        cnt = config_counts[cid]
                        if cnt >= 7:
                            valid = False
                            break
                        score += (7 - cnt)
                    if valid and score > best_score:
                        best_score = score
                        best_t = t_idx
                        
                if best_t == -1:
                    break # No valid triples left
                
                # Add best_t
                present[best_t] = True
                current_set.add(best_t)
                remaining.remove(best_t)
                
                for cid in triple_to_configs[best_t]:
                    config_counts[cid] += 1
                    
                # Note: We don't remove invalid triples from 'remaining' immediately, 
                # we just skip them in the next iteration's scan.
                # This is O(N^2 * Degree) total.
                
        return len(current_set), list(current_set)
        
    best_count = 0
    best_set = []
    
    # Try multiple strategies
    # 1. Random restarts
    for i in range(20):
        count, s = run_greedy("random", seed=i)
        if count > best_count:
            best_count = count
            best_set = s
        if time.time() > deadline - 5:
            break
            
    # 2. Slack-based greedy (deterministic)
    count, s = run_greedy("slack", seed=0)
    if count > best_count:
        best_count = count
        best_set = s
        
    # 3. Hybrid: Random start, then local search?
    # The seed program did local search. The prompt asks for a DIFFERENT strategy.
    # Greedy Set Packing is structurally different from WalkSAT/Annealing.
    # However, to beat the seed (2.29), we might need a bit of local search on the greedy result.
    # But the directive says "Make a STRUCTURAL change... A first try in a new direction may score worse".
    # I will stick to pure Greedy to be safe on the "Different Strategy" constraint, 
    # but I will add a simple "1-swap" local search at the end if time permits, 
    # as it is often considered part of the "Greedy + Local Search" family, 
    # which is distinct from "Stochastic Local Search from Scratch".
    # Actually, the seed ALREADY has local search.
    # So I should avoid adding the SAME local search.
    
    # Let's try a "Weighted Random" greedy where probability is proportional to slack.
    # This is a different sampling distribution.
    
    def run_weighted_greedy(seed=0):
        present[:] = False
        config_counts[:] = 0
        current_set = set()
        rng = random.Random(seed)
        
        remaining = list(range(num_triples))
        
        while remaining:
            if time.time() > deadline - 2:
                break
            
            # Calculate scores for all remaining
            scores = []
            valid_indices = []
            for i, t_idx in enumerate(remaining):
                score = 0
                valid = True
                for cid in triple_to_configs[t_idx]:
                    cnt = config_counts[cid]
                    if cnt >= 7:
                        valid = False
                        break
                    score += (7 - cnt)
                if valid:
                    scores.append(score)
                    valid_indices.append(i)
                    
            if not valid_indices:
                break
                
            # Pick one with probability proportional to score^2 (sharper)
            # Or just max score (greedy)
            # Let's try max score but with random tie-breaking
            max_score = max(scores)
            candidates = [valid_indices[j] for j in range(len(scores)) if scores[j] == max_score]
            chosen_idx_in_remaining = rng.choice(candidates)
            t_idx = remaining[chosen_idx_in_remaining]
            
            present[t_idx] = True
            current_set.add(t_idx)
            remaining.pop(chosen_idx_in_remaining)
            
            for cid in triple_to_configs[t_idx]:
                config_counts[cid] += 1
                
        return len(current_set), list(current_set)
        
    # Run weighted greedy
    for i in range(10):
        count, s = run_weighted_greedy(seed=100+i)
        if count > best_count:
            best_count = count
            best_set = s
        if time.time() > deadline - 5:
            break
            
    result = [all_triples[i] for i in best_set]
    
    return result
