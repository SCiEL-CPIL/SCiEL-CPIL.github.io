import itertools
import random
import time

def construct():
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute all K(2,2,2)s
    k_list = []
    triple_to_ks = [[] for _ in range(num_triples)]
    k_to_triples = []
    
    matchings = [
        ((0,1),(2,3),(4,5)),
        ((0,1),(2,4),(3,5)),
        ((0,1),(2,5),(3,4)),
        ((0,2),(1,3),(4,5)),
        ((0,2),(1,4),(3,5)),
        ((0,2),(1,5),(3,4)),
        ((0,3),(1,2),(4,5)),
        ((0,3),(1,4),(2,5)),
        ((0,3),(1,5),(2,4)),
        ((0,4),(1,2),(3,5)),
        ((0,4),(1,3),(2,5)),
        ((0,4),(1,5),(2,3)),
        ((0,5),(1,2),(3,4)),
        ((0,5),(1,3),(2,4)),
        ((0,5),(1,4),(2,3))
    ]
    
    for subset in itertools.combinations(range(n), 6):
        for m in matchings:
            P1 = (subset[m[0][0]], subset[m[0][1]])
            P2 = (subset[m[1][0]], subset[m[1][1]])
            P3 = (subset[m[2][0]], subset[m[2][1]])
            
            k_triples = []
            for x in P1:
                for y in P2:
                    for z in P3:
                        t = tuple(sorted((x, y, z)))
                        k_triples.append(t)
            
            k_triple_indices = [triple_to_idx[t] for t in k_triples]
            k_list.append(k_triple_indices)
            k_to_triples.append(k_triple_indices)
            
            for idx in k_triple_indices:
                triple_to_ks[idx].append(len(k_list) - 1)

    num_ks = len(k_list)
    max_score = 286 # Max Ks per triple

    def run_greedy_lsf():
        counts = [0] * num_ks
        blocked = [0] * num_triples
        used = [False] * num_triples
        scores = [0] * num_triples
        
        # Buckets: buckets[s] is a set of triple indices with score s
        # Max score is 286, so size 287
        buckets = [set() for _ in range(max_score + 1)]
        
        # Initialize buckets
        for i in range(num_triples):
            buckets[0].add(i)
            
        current_min_score = 0
        result = []
        
        while True:
            # Find the lowest score bucket that has available, unblocked triples
            while current_min_score <= max_score and not buckets[current_min_score]:
                current_min_score += 1
            
            if current_min_score > max_score:
                break
                
            # Pick a random triple from the bucket
            # Convert set to list for random selection? Or just pop?
            # If we pop, we lose the element. That's fine.
            # But we need to be careful: if we pop, it's removed.
            # We need to ensure it's not blocked or used.
            # Since we only add to buckets if available/unblocked, and remove when blocked/used,
            # the sets should only contain valid candidates.
            
            # However, a triple might be in the bucket but get blocked by a previous step in the same iteration?
            # No, we process one triple at a time.
            # But when we update scores for neighbors, we move them.
            # So the sets are consistent.
            
            # Pick one
            candidate_set = buckets[current_min_score]
            if not candidate_set:
                continue
            chosen_idx = random.choice(list(candidate_set))
            candidate_set.remove(chosen_idx)
            
            result.append(all_triples[chosen_idx])
            used[chosen_idx] = True
            
            # Update counts and scores
            for k_idx in triple_to_ks[chosen_idx]:
                old_count = counts[k_idx]
                counts[k_idx] += 1
                
                # If this K was empty (0), it becomes active (1)
                if old_count == 0:
                    for t_idx in k_to_triples[k_idx]:
                        if not used[t_idx] and blocked[t_idx] == 0:
                            # Move from old score to new score
                            old_s = scores[t_idx]
                            new_s = old_s + 1
                            scores[t_idx] = new_s
                            buckets[old_s].discard(t_idx)
                            buckets[new_s].add(t_idx)
                
                # If this K becomes saturated (7), block the remaining triple
                if counts[k_idx] == 7:
                    for t_idx in k_to_triples[k_idx]:
                        if not used[t_idx] and blocked[t_idx] == 0:
                            blocked[t_idx] += 1
                            old_s = scores[t_idx]
                            buckets[old_s].discard(t_idx)
                            break

        return result

    best_result = []
    start_time = time.time()
    
    # Run as many restarts as possible
    while time.time() - start_time < 110:
        res = run_greedy_lsf()
        if len(res) > len(best_result):
            best_result = res

    return best_result
