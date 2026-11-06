import math
import time
import sys

def construct():
    """
    Construct a nonempty subset of [1, 255] containing 255,
    with gcd(a,b) > 1 for all distinct pairs. Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # 255 = 3 * 5 * 17
    # Elements in the set must share a common prime factor with 255,
    # OR be paired such that every pair has gcd > 1.
    #
    # Strategy: Use a clique-based approach.
    # 1. Identify all numbers in [1, 255] that share a factor with 255 (i.e., divisible by 3, 5, or 17).
    # 2. From these, find the largest subset where every pair has gcd > 1.
    #
    # Note: If two numbers are both divisible by 3, their gcd is at least 3.
    # If two numbers are both divisible by 5, their gcd is at least 5.
    # But a number divisible by 3 and a number divisible by 5 (but not 3 or 5) might have gcd 1.
    #
    # So we need to be careful. Let's group by prime factors of 255: {3, 5, 17}.
    # Any number in the set must be divisible by at least one of {3, 5, 17} to have gcd > 1 with 255.
    # Wait, the condition is gcd(a,b) > 1 for ALL distinct pairs in the set.
    # Since 255 is in the set, every other element a must satisfy gcd(a, 255) > 1.
    # So every element must be divisible by 3, 5, or 17.
    #
    # Now, among these candidates, we need gcd(a,b) > 1 for all pairs a,b.
    #
    # Let A = {x in [1,255] : 3|x}, B = {x in [1,255] : 5|x}, C = {x in [1,255] : 17|x}.
    # Candidates = A ∪ B ∪ C.
    #
    # If we pick all numbers divisible by 3, then any two have gcd >= 3. Size = floor(255/3) = 85.
    # If we pick all numbers divisible by 5, any two have gcd >= 5. Size = floor(255/5) = 51.
    # If we pick all numbers divisible by 17, any two have gcd >= 17. Size = floor(255/17) = 15.
    #
    # Can we do better by mixing?
    # Suppose we pick some numbers divisible by 3 and some by 5.
    # For any a (div by 3) and b (div by 5) in the set, we need gcd(a,b) > 1.
    # This means either a is also divisible by 5, or b is also divisible by 3, or they share another prime.
    #
    # Let's think about it differently. We want the largest clique in the graph where vertices are [1,255]
    # and edges connect pairs with gcd > 1. But we must include 255.
    # Actually, since 255 must be included, and 255's prime factors are {3,5,17},
    # every other vertex in the clique must be adjacent to 255, i.e., share a factor 3, 5, or 17.
    #
    # So the problem reduces to: find the largest clique in the graph induced by
    # V = {x in [1,255] : gcd(x,255) > 1} ∪ {255}, with the constraint that 255 is in the clique.
    #
    # Since 255 is connected to all of V, we just need the largest clique in the subgraph on V
    # that is also connected to 255 (which is all of V). So we need the largest clique in the graph
    # where vertices are {x in [1,255] : gcd(x,255) > 1} and edges are gcd > 1.
    #
    # Let's analyze the structure.
    # Let S3 = {x : 3|x, 1<=x<=255}, S5 = {x : 5|x, 1<=x<=255}, S17 = {x : 17|x, 1<=x<=255}.
    # 
    # Key insight: If we take all of S3, that's a clique of size 85.
    # Can we add elements from S5 or S17 that are not in S3?
    # An element x in S5 \ S3 has gcd(x, y) > 1 for all y in S3 only if... 
    # Actually, for x in S5\S3 and y in S3, gcd(x,y) > 1 requires that they share a prime factor.
    # x is divisible by 5 but not 3. y is divisible by 3.
    # If y is only divisible by 3 (and not 5 or 17 or any common factor with x), then gcd(x,y) could be 1.
    # For example, x=5, y=3: gcd(5,3)=1. So we can't have both 5 and 3 in the set.
    #
    # So the set must be "monochromatic" in some sense? Not necessarily.
    # But if we include any number from S3 that is not divisible by 5 or 17, then all other numbers
    # in the set must be divisible by 3 (to share the factor 3) OR share another factor with this number.
    #
    # This is getting complex. Let's use a greedy or exact algorithm for maximum clique on this specific graph.
    # The graph has at most 85+51+15 - overlaps = let's compute.
    # |S3| = 85, |S5| = 51, |S17| = 15.
    # |S3∩S5| = |S15| = 17, |S3∩S17| = |S51| = 5, |S5∩S17| = |S85| = 3.
    # |S3∩S5∩S17| = |S255| = 1.
    # Union = 85 + 51 + 15 - 17 - 5 - 3 + 1 = 127.
    # So at most 127 vertices. Maximum clique in a graph of 127 vertices is feasible with Bron-Kerbosch.
    
    # Let's implement a maximum clique finder using Bron-Kerbosch with pivot.
    
    # First, build the candidate set
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, 255) > 1:
            candidates.append(x)
    
    n = len(candidates)
    
    # Build adjacency: adj[i] is a set of j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Bron-Kerbosch with pivot
    def bron_kerbosch(R, P, X):
        if time.time() > deadline:
            return []
        
        if not P and not X:
            return R
        
        # Choose pivot u from P ∪ X
        U = P | X
        if not U:
            return R
            
        # Pick the vertex with maximum degree in U
        pivot = max(U, key=lambda u: len(adj[u] & P))
        
        # For each v in P \ N(pivot)
        candidates_for_recursion = P - adj[pivot]
        
        best = []
        for v in list(candidates_for_recursion):
            if time.time() > deadline:
                break
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            result = bron_kerbosch(new_R, new_P, new_X)
            if len(result) > len(best):
                best = result
        
        return best
    
    # Initialize
    P = set(range(n))
    X = set()
    R = set()
    
    # Find maximum clique
    max_clique = bron_kerbosch(R, P, X)
    
    # Convert back to values
    result_set = set(candidates[i] for i in max_clique)
    
    # Ensure 255 is in the result
    # 255 should be in candidates since gcd(255,255)=255>1
    if 255 not in result_set:
        # This shouldn't happen if 255 is the best choice, but let's be safe
        # Actually, if 255 isn't in the max clique found, we might need to force it.
        # But since we want to include 255, let's verify.
        # The max clique in the induced subgraph might not include 255 if there's a larger clique without it.
        # But we MUST include 255. So we need to find the max clique that includes 255.
        
        # Find index of 255
        idx_255 = candidates.index(255)
        
        # Redo with constraint that 255 is in the clique
        # R = {idx_255}, P = adj[idx_255], X = empty
        R = {idx_255}
        P = adj[idx_255]
        X = set()
        max_clique_with_255 = bron_kerbosch(R, P, X)
        result_set = set(candidates[i] for i in max_clique_with_255)
    
    # Double-check validity
    res_list = list(result_set)
    if not res_list:
        return [255]
    
    for i in range(len(res_list)):
        for j in range(i+1, len(res_list)):
            if math.gcd(res_list[i], res_list[j]) == 1:
                # Fallback to all multiples of 3
                return list(range(3, 256, 3))
    
    if 255 not in res_list:
        return list(range(3, 256, 3))
    
    return res_list

# For quick testing
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Contains 255: {255 in result}")
    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                valid = False
                break
    print(f"Valid: {valid}")
