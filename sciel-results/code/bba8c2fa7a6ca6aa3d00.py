import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # Specific case: q=3, n=6, radius=1
    # Target: K_3(6,1) <= 73 (best known), lower bound 71.
    # Strategy:
    # 1. Construct a maximal independent set in the distance-2 graph.
    #    This ensures no two codewords are at distance 2.
    #    This creates a "spread out" code where each codeword's ball of radius 1
    #    overlaps less with others in a specific way, potentially allowing more deletions.
    # 2. Use Simulated Annealing with 1-delete and 2-swap moves to minimize size.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: direct sum of tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    # Precompute neighbors and distance-2 pairs
    # Words are 0..728
    def num_to_vec(num, length=6):
        vec = [0] * length
        for i in range(length):
            vec[i] = num % 3
            num //= 3
        return vec

    def vec_to_num(vec):
        num = 0
        for i in range(len(vec)):
            num += vec[i] * (3 ** i)
        return num

    # cover_set[i]: list of words covered by codeword i (distance <= 1)
    cover_set = []
    dist2_pairs = []  # list of (i, j) such that dist(i, j) == 2
    
    # Precompute vectors for all 729 words
    all_vecs = [num_to_vec(i) for i in range(729)]
    
    for i in range(729):
        v = all_vecs[i]
        s = [i]
        for pos in range(6):
            orig = v[pos]
            for new_val in range(3):
                if new_val != orig:
                    v[pos] = new_val
                    s.append(vec_to_num(v))
            v[pos] = orig
        cover_set.append(s)
        
    # Find distance-2 pairs for independent set construction
    # dist(i, j) == 2 means they differ in exactly 2 positions
    # We can compute this efficiently or just check during greedy independent set.
    # For 729 nodes, checking all pairs is 729*729/2 ~ 265k checks.
    # Each check is Hamming distance.
    # Let's precompute a boolean matrix or list of lists for dist2.
    # dist2_adj[i] = set of j such that dist(i, j) == 2
    
    dist2_adj = [set() for _ in range(729)]
    for i in range(729):
        vi = all_vecs[i]
        for j in range(i + 1, 729):
            vj = all_vecs[j]
            dist = 0
            for k in range(6):
                if vi[k] != vj[k]:
                    dist += 1
                    if dist > 2:
                        break
            if dist == 2:
                dist2_adj[i].add(j)
                dist2_adj[j].add(i)

    # Phase 1: Greedy Maximal Independent Set in Distance-2 Graph
    # We want a set S such that no two elements in S are at distance 2.
    # And S covers all 729 words with radius 1.
    # Greedy: Pick nodes that cover the most uncovered words, but only if they don't conflict with existing picks (dist 2).
    # Actually, a simple maximal independent set might not cover everything.
    # We need a COVER that is an independent set in the distance-2 graph?
    # No, the constraint is just that we *want* to avoid distance 2 pairs to make deletion easier.
    # If we pick a maximal independent set, it might not cover the space.
    # Let's use a heuristic:
    # 1. Start with empty set.
    # 2. While uncovered exists:
    #    a. Find candidate c that covers max uncovered words.
    #    b. If c is at distance 2 from any selected node, we have a choice:
    #       - Add it anyway (break independence) if necessary for coverage.
    #       - Or try to find another node that covers similar words but is independent.
    #    
    # Better approach:
    # 1. Generate a random maximal independent set (MIS) of the distance-2 graph.
    #    This set is "spread out".
    # 2. Check if it covers the space. If not, add nodes greedily to cover gaps.
    # 3. Optimize with local search.
    
    # Let's try a direct greedy that penalizes distance-2 conflicts.
    # Score(c) = num_uncovered_covered(c) - lambda * num_conflicts(c)
    
    uncovered = np.ones(729, dtype=bool)
    selected = set()
    
    # Initial greedy with penalty
    lambda_penalty = 1.0
    
    while np.any(uncovered) and time.time() < deadline - 10:
        best_c = -1
        best_score = -1
        # Optimization: only consider candidates that cover at least 1 uncovered word
        # We can precompute which candidates are worth looking at?
        # For now, iterate all.
        for c in range(729):
            if c in selected:
                continue
            # Gain
            gain = np.sum(uncovered[cover_set[c]])
            if gain == 0:
                continue
            # Conflicts: how many selected nodes are at distance 2 from c
            conflicts = sum(1 for s in selected if c in dist2_adj[s])
            score = gain - lambda_penalty * conflicts
            if score > best_score:
                best_score = score
                best_c = c
        
        if best_c == -1:
            break
            
        selected.add(best_c)
        uncovered[cover_set[best_c]] = False
        
    if not np.any(uncovered):
        current_set = set(selected)
    else:
        # Fallback if greedy failed to cover (should be rare with penalty 1.0)
        # Just add whatever covers the most uncovered
        while np.any(uncovered):
            best_c = -1
            max_gain = 0
            for c in range(729):
                if c in selected: continue
                gain = np.sum(uncovered[cover_set[c]])
                if gain > max_gain:
                    max_gain = gain
                    best_c = c
            if best_c == -1: break
            selected.add(best_c)
            uncovered[cover_set[best_c]] = False
        current_set = set(selected)
        
    best_size = len(current_set)
    best_code_set = set(current_set)
    
    # Phase 2: Simulated Annealing / Local Search
    # Objective: Minimize |S|
    # Moves:
    # 1. Delete c if valid (all words still covered).
    # 2. Swap c_out, c_in if it leads to a valid state with same size (neutral) or allows future deletion.
    #    To be effective, we need to accept neutral moves to escape local minima.
    #    We use a "potential" function: P = sum(1 if cover_count[w]==1 else 0)
    #    Minimizing P (exposed words) makes deletion easier.
    
    cover_count = np.zeros(729, dtype=np.int32)
    
    def update_counts(S):
        cover_count[:] = 0
        for c in S:
            for w in cover_set[c]:
                cover_count[w] += 1
                
    update_counts(current_set)
    
    def get_potential():
        # Number of words with exactly 1 cover
        return np.sum(cover_count == 1)
    
    current_potential = get_potential()
    
    # Annealing parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute list of all words for random access? No, 0..728 is fine.
    
    step = 0
    while time.time() < deadline - 2 and T > min_T:
        step += 1
        
        # 1. Try Deletion
        # Pick a random candidate for deletion
        # Prioritize codewords that are "redundant" (high min count in their cover set)
        candidates = list(current_set)
        random.shuffle(candidates)
        
        deleted = False
        for c in candidates[:20]: # Check top 20 random
            # Check if we can remove c
            # If min(cover_count[w]) >= 2 for all w in cover_set[c], we can remove c.
            # Actually, if cover_count[w] >= 2 for all w in cover_set[c], removing c leaves count >= 1.
            # Words not in cover_set[c] are unaffected.
            can_remove = True
            for w in cover_set[c]:
                if cover_count[w] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Perform deletion
                current_set.remove(c)
                for w in cover_set[c]:
                    cover_count[w] -= 1
                best_size = len(current_set)
                best_code_set = set(current_set)
                deleted = True
                break
        
        if deleted:
            continue # Restart deletion check
        
        # 2. Try Swap
        # Pick c_out from current_set
        # Pick c_in from not in current_set
        # Accept if:
        #   a) It allows a deletion (not checked immediately, but potential improves)
        #   b) It reduces potential (exposed words)
        #   c) With probability exp(-delta/T), accept worse potential (to escape)
        
        c_out = random.choice(list(current_set))
        # Choose c_in
        # Heuristic: Choose c_in that covers many exposed words
        # Exposed words are those with cover_count == 1
        exposed_mask = (cover_count == 1)
        
        # Find best c_in for this c_out
        # We want c_in to cover exposed words that c_out covers?
        # Actually, if we remove c_out, words in cover_set[c_out] lose 1 count.
        # Those with count 1 become 0 (invalid).
        # So c_in must cover all words in cover_set[c_out] that have count 1.
        
        critical_words = [w for w in cover_set[c_out] if cover_count[w] == 1]
        
        if not critical_words:
            # c_out is redundant, should have been deleted. Skip or force delete?
            # Maybe just swap to explore.
            pass
        else:
            # Find c_in that covers all critical_words
            # And ideally covers other exposed words too.
            
            # Efficient check:
            # c_in is valid if set(critical_words) is subset of cover_set[c_in]
            
            # We can't check all 729 c_in for every c_out quickly in Python loops.
            # Let's sample a few c_in.
            
            swap_found = False
            for _ in range(10):
                c_in = random.randint(0, 728)
                if c_in in current_set:
                    continue
                
                # Check if c_in covers all critical_words
                # Optimization: Precompute cover_set as sets?
                # cover_set is lists. 
                # Let's check membership.
                valid_in = True
                for w in critical_words:
                    if w not in cover_set[c_in]: # O(1) if set, O(19) if list
                        valid_in = False
                        break
                if not valid_in:
                    continue
                
                # Calculate change in potential
                # Remove c_out, add c_in.
                # New cover_count:
                # For w in cover_set[c_out]: count -= 1
                # For w in cover_set[c_in]: count += 1
                # Potential change = sum(1 if new_count==1) - sum(1 if old_count==1)
                
                # This is expensive to compute exactly for every candidate.
                # Approximation:
                # If c_in covers more exposed words (that are NOT in critical_words, i.e. count 1 but not affected by c_out removal? No, if count 1 and in cover_set[c_out], it becomes 0 unless covered by c_in. If covered by c_in, it becomes 1 again. So net change 0 for those.
                # Words in cover_set[c_in] but NOT in cover_set[c_out]:
                #   If count was 0 (impossible, covered by others? No, if count 0 it's uncovered, but we maintain validity. Wait, if we remove c_out, some words might become 0 if not covered by c_in. We ensured critical_words are covered. Other words in cover_set[c_out] have count >= 2, so they become >= 1. So validity is maintained.
                #   So, for w in cover_set[c_in] \ cover_set[c_out]:
                #       old_count = cover_count[w]
                #       new_count = old_count + 1
                #       Change in potential:
                #       If old_count == 1, new_count == 2 -> Potential -1 (Good)
                #       If old_count == 2, new_count == 3 -> Potential 0
                #       If old_count >= 3, Potential 0
                #   For w in cover_set[c_out] \ cover_set[c_in]:
                #       old_count = cover_count[w]
                #       new_count = old_count - 1
                #       Change in potential:
                #       If old_count == 2, new_count == 1 -> Potential +1 (Bad)
                #       If old_count == 1, new_count == 0 -> Invalid (But we checked critical_words, so this shouldn't happen for w in critical_words. For w not in critical_words, count >= 2, so new_count >= 1. If old_count==2, new_count==1 -> +1)
                #       If old_count >= 3, new_count >= 2 -> Potential 0
                
                # Let's just compute the new potential for the top few candidates.
                # Since we are in a tight loop, let's do a fast heuristic check.
                
                # Heuristic:
                # gain = number of words in cover_set[c_in] \ cover_set[c_out] with count == 1
                # loss = number of words in cover_set[c_out] \ cover_set[c_in] with count == 2
                # delta_potential = loss - gain
                
                # Fast approximation using numpy?
                # It's hard to do set difference fast with lists.
                
                # Let's just try the swap and compute potential.
                # Save state
                old_counts = cover_count.copy()
                
                for w in cover_set[c_out]:
                    cover_count[w] -= 1
                for w in cover_set[c_in]:
                    cover_count[w] += 1
                
                new_potential = np.sum(cover_count == 1)
                delta_p = new_potential - current_potential
                
                if delta_p < 0:
                    # Accept
                    current_set.remove(c_out)
                    current_set.add(c_in)
                    current_potential = new_potential
                    swap_found = True
                    break
                else:
                    # Revert
                    cover_count[:] = old_counts
                    # Accept with probability exp(-delta_p / T) ?
                    # delta_p is small integer.
                    if random.random() < np.exp(-delta_p / T):
                        current_set.remove(c_out)
                        current_set.add(c_in)
                        current_potential = new_potential
                        swap_found = True
                        break
            
        if not swap_found:
            # No good swap found, cool down
            T *= cooling
            
    # Final verification and cleanup
    # Try a final pass of deletions
    update_counts(best_code_set)
    improved = True
    while improved:
        improved = False
        candidates = list(best_code_set)
        random.shuffle(candidates)
        for c in candidates:
            can_remove = True
            for w in cover_set[c]:
                if cover_count[w] == 1:
                    can_remove = False
                    break
            if can_remove:
                best_code_set.remove(c)
                for w in cover_set[c]:
                    cover_count[w] -= 1
                improved = True
                break

    # Return best found
    if best_code_set:
        # Final check
        final_counts = np.zeros(729, dtype=np.int32)
        for c in best_code_set:
            for w in cover_set[c]:
                final_counts[w] += 1
        if np.min(final_counts) >= 1:
            return [num_to_vec(c) for c in best_code_set]
            
    # Fallback
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
