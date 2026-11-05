"""Program for the suite task: maximize subset of [1,100] with no subset summing to 200."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    # We restrict the candidate set to [51, 100].
    # Reasoning:
    # 1. No two distinct elements in [1, 100] sum to 200 (max is 99+100=199).
    # 2. Therefore, any forbidden subset must have size >= 3.
    # 3. For elements in [51, 100], the maximum size of a subset summing to 200 is 3 (since 4*51=204>200).
    # 4. For smaller elements, subsets of size 4, 5, etc., can sum to 200, creating a denser conflict graph.
    # 5. Maximizing cardinality is typically easier in the "large number" region where conflicts are sparser (limited to triples).
    
    candidates = list(range(51, 101))
    
    # Pre-compute all triples that sum to 200
    # This defines the "conflict hypergraph"
    conflict_triples = []
    for a, b, c in itertools.combinations(candidates, 3):
        if a + b + c == 200:
            conflict_triples.append((a, b, c))
            
    # Calculate the "degree" (number of conflicts) for each candidate
    # A candidate v is in a conflict if it is part of a triple summing to 200
    degrees = {v: 0 for v in candidates}
    for a, b, c in conflict_triples:
        degrees[a] += 1
        degrees[b] += 1
        degrees[c] += 1
        
    # Sort candidates by increasing degree (fewest conflicts first)
    # This is the "greedy heuristic that prioritizes vertices with the fewest conflicts"
    sorted_candidates = sorted(candidates, key=lambda x: degrees[x])
    
    result = []
    
    # To efficiently check if adding a new element v creates a forbidden sum (triple),
    # we check if there exist two elements a, b in 'result' such that a + b + v = 200.
    # i.e., a + b = 200 - v.
    
    for v in sorted_candidates:
        target_sum = 200 - v
        is_conflict = False
        # Check all pairs in current result
        # Optimization: We can break early if found
        # Since result size is small, O(N^2) is fine.
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if result[i] + result[j] == target_sum:
                    is_conflict = True
                    break
            if is_conflict:
                break
        
        if not is_conflict:
            result.append(v)
            
    # Ensure nonempty
    if not result:
        return [100]
        
    return result
# EVOLVE-BLOCK-END
