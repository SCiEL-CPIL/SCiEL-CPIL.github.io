import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    
    # 255 = 3 * 5 * 17
    # We need a subset of [1, 255] containing 255 with gcd(a,b) > 1 for all pairs.
    # Strategy: Find a large clique in the graph where edges connect numbers with gcd > 1.
    # Key insight: All multiples of 3 in [1, 255] form a valid set (size 85).
    # We can try to do better by finding a larger clique.
    
    # Generate all candidates: numbers in [1, 255] that share a factor with 255
    # Actually, any number in [1, 255] can be in the set as long as it's pairwise non-coprime
    # with all others. But 255 must be in the set.
    # So every other element must share a factor with 255 (i.e., with 3, 5, or 17).
    
    candidates = [x for x in range(1, 256) if math.gcd(x, 255) > 1]
    # 255 is in candidates since gcd(255, 255) = 255 > 1
    n = len(candidates)
    
    # Precompute gcd matrix for candidates
    # candidates[i] and candidates[j] are adjacent if gcd(candidates[i], candidates[j]) > 1
    # We want to find a maximum clique containing 255.
    
    # Since 255 must be in the set, and 255 = 3*5*17, every other element in the set
    # must share a factor with 255, which is already guaranteed by our candidate filter.
    # But we also need pairwise gcd > 1 among all selected elements.
    
    # Let's use a greedy approach with local search, or a more systematic clique finding.
    # For n=127, we can use a branch-and-bound or greedy with restarts.
    
    # Map values to indices
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    idx_255 = val_to_idx[255]
    
    # Precompute adjacency: adj[i] is a set of j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Greedy: start with 255, then greedily add vertices that are connected to all current members
    # But this might not be optimal. Let's try a better approach.
    
    # Since the graph is dense (most pairs of numbers sharing factors with 255 are connected),
    # let's try to find a large clique using a heuristic.
    
    # Approach: Start with all multiples of 3 (which is a valid clique of size 85).
    # Then try to add other vertices that are connected to all members.
    
    # Initial clique: all multiples of 3 in [1, 255]
    clique = [x for x in candidates if x % 3 == 0]
    clique_set = set(clique)
    clique_indices = {val_to_idx[x] for x in clique}
    
    # Try to add vertices not in the clique that are connected to all clique members
    for x in candidates:
        if x in clique_set:
            continue
        if time.time() > deadline - 5.0:
            break
        # Check if x is connected to all members of the clique
        compatible = True
        for y in clique:
            if math.gcd(x, y) == 1:
                compatible = False
                break
        if compatible:
            clique.append(x)
            clique_set.add(x)
            clique_indices.add(val_to_idx[x])
    
    # Now try local search: try to swap out one vertex for two, or add more
    # Actually, let's try a more systematic approach: maximum clique in the subgraph
    
    # Let's try a branch and bound for maximum clique, starting from the current best
    best_clique = list(clique)
    best_size = len(clique)
    
    # Try to improve by considering all candidates
    # Use a simple local search: for each candidate not in best, see if we can add it
    # by removing some vertices
    
    # A better approach: since we have a decent starting point, let's try to find
    # if there's a larger clique by examining the structure.
    
    # The multiples of 3 give 85. Can we do better?
    # Consider numbers divisible by 3: 85 numbers.
    # Numbers divisible by 5 but not 3: 51 - 17 = 34 numbers (multiples of 5 in [1,255] that aren't multiples of 3)
    # Wait: multiples of 5 in [1,255]: 51. Multiples of 15: 17. So 51-17=34 multiples of 5 not divisible by 3.
    # These 34 numbers don't share factor 3 with each other necessarily... actually they all share factor 5.
    # But do they share factors with the multiples of 3? Not necessarily.
    
    # Let's think differently. The maximum clique might be all numbers divisible by 3, 
    # plus some additional numbers that share factors with all multiples of 3.
    # For a number x to be connected to ALL multiples of 3, x must share a factor with every multiple of 3.
    # In particular, x must share a factor with 3, 6, 9, ..., 255.
    # x must share a factor with 3 (so x is divisible by 3, or x shares a factor with 3, i.e., x is divisible by 3).
    # Wait, gcd(x, 3) > 1 means x is divisible by 3.
    # So any number not divisible by 3 has gcd(x, 3) = 1, meaning it's NOT connected to 3.
    # Since 3 is in our clique (3 is a multiple of 3), any number not divisible by 3 cannot be added.
    
    # Therefore, the maximum clique containing 3 (and 255) is exactly the set of all multiples of 3!
    # Wait, that's not right. 3 is in the set, and for any other element y in the set, gcd(3, y) > 1,
    # which means y must be divisible by 3. So yes, all elements must be divisible by 3.
    # Thus the maximum set containing 3 and 255 with pairwise gcd > 1 is exactly the multiples of 3.
    
    # But wait, do we HAVE to include 3? No, we just need to include 255.
    # The set doesn't need to include 3. It needs to include 255, and all pairs must have gcd > 1.
    
    # So we don't need 3 in the set. We need 255 in the set, and every other element must share
    # a factor with 255 (i.e., with 3, 5, or 17), AND every pair must share a factor.
    
    # Let me reconsider. The set must contain 255. Every other element must have gcd > 1 with 255.
    # And every pair of elements must have gcd > 1.
    
    # Multiples of 3: size 85. This is valid since 255 is a multiple of 3.
    # Can we get more than 85?
    
    # Consider: all numbers divisible by 3 OR (divisible by 5 AND share a factor with all multiples of 3 in the set).
    # This is getting complex. Let's just run a proper maximum clique algorithm.
    
    # For n=127, a good heuristic or exact algorithm should work within time limits.
    # Let's use a greedy + local search approach.
    
    # Actually, let me reconsider the structure. 255 = 3*5*17.
    # The set must contain 255.
    # Let S be the set. For all a, b in S, a != b, gcd(a,b) > 1.
    # 
    # Key observation: If we take all numbers divisible by 3, that's 85 numbers, all pairwise share factor 3.
    # If we take all numbers divisible by 5, that's 51 numbers.
    # If we take all numbers divisible by 17, that's 15 numbers.
    #
    # The union of multiples of 3 and multiples of 5 is NOT a valid set because a multiple of 3 not divisible by 5
    # and a multiple of 5 not divisible by 3 might be coprime.
    #
    # For example, 3 and 5 are coprime. So {3, 5} is not valid.
    #
    # What if we take all multiples of 3, plus some multiples of 5 that share factors with all multiples of 3?
    # A multiple of 5 (not divisible by 3) needs to share a factor with every multiple of 3 in the set.
    # In particular, it needs to share a factor with 3. But gcd(5k, 3) = gcd(k, 3) * ... 
    # Actually gcd(5k, 3) = 1 if 3 doesn't divide 5k, i.e., if 3 doesn't divide k.
    # So for 5k to share a factor with 3, we need 3 | 5k, which means 3 | k.
    # But then 5k is divisible by 3, so it's already in our multiples of 3 set.
    #
    # So no multiple of 5 that is NOT divisible by 3 can be added to a set containing 3.
    # But we don't have to include 3!
    
    # Let's think about this more carefully. We want to maximize |S| where 255 in S and pairwise gcd > 1.
    
    # Strategy: Choose a "core" prime p, and take all multiples of p. Size = floor(255/p).
    # p=3: 85, p=5: 51, p=17: 15.
    # So multiples of 3 gives 85.
    
    # Can we mix? For instance, take all multiples of 3, and then add some numbers that are not multiples of 3
    # but share factors with all the multiples of 3 we've chosen?
    
    # If we choose NOT to include 3, then we might be able to add some non-multiples of 3.
    # For example, if we exclude 3 from our set, then a number like 5 could potentially be included
    # if it shares factors with all other elements.
    
    # This is a complex optimization. Let's use a computational approach.
    
    # Given the time constraint, let's use a greedy algorithm with restarts and local search.
    
    # First, let's establish a baseline: multiples of 3 gives 85.
    baseline = [x for x in range(1, 256) if x % 3 == 0]
    best_size = len(baseline)
    best_set = set(baseline)
    
    # Try other strategies:
    # 1. All multiples of 3 (85)
    # 2. Try to find a larger set using search
    
    # Let's try a simple local search starting from multiples of 3
    current_set = set(baseline)
    current_list = list(current_set)
    
    # Try adding elements one by one
    for x in candidates:
        if x in current_set:
            continue
        if time.time() > deadline - 10.0:
            break
        # Check if x can be added
        can_add = True
        for y in current_set:
            if math.gcd(x, y) == 1:
                can_add = False
                break
        if can_add:
            current_set.add(x)
    
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = set(current_set)
    
    # Now try local search: for each element in the set, try removing it and adding two others
    # This is complex, so let's try a simpler approach
    
    # Try starting from different initial sets
    # Initial set: all numbers sharing a factor with 255 (all candidates)
    # Then greedily remove elements that cause conflicts
    
    # Actually, let's try a maximum clique approach using a simple heuristic
    # Order candidates by degree (descending) and greedily build a clique
    
    # Compute degrees
    degrees = [len(adj[i]) for i in range(n)]
    
    # Sort candidates by degree descending
    sorted_indices = sorted(range(n), key=lambda i: -degrees[i])
    
    # Greedy clique construction
    greedy_clique = set()
    for i in sorted_indices:
        if time.time() > deadline - 10.0:
            break
        # Check if candidate[i] is connected to all in greedy_clique
        val = candidates[i]
        if val == 255 or (greedy_clique and all(math.gcd(val, candidates[j]) > 1 for j in greedy_clique)):
            if not greedy_clique or all(math.gcd(val, candidates[j]) > 1 for j in greedy_clique):
                greedy_clique.add(i)
    
    # Ensure 255 is in the clique
    if idx_255 not in greedy_clique:
        # Remove conflicting elements and add 255
        greedy_clique.discard(idx_255)
        # Rebuild
        greedy_clique = {idx_255}
        for i in sorted_indices:
            if i == idx_255:
                continue
            if time.time() > deadline - 10.0:
                break
            if all(math.gcd(candidates[i], candidates[j]) > 1 for j in greedy_clique):
                greedy_clique.add(i)
    
    greedy_size = len(greedy_clique)
    if greedy_size > best_size:
        best_size = greedy_size
        best_set = {candidates[i] for i in greedy_clique}
    
    # Try multiple random restarts with greedy + local search
    import random
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Random initial set: start with 255 and randomly add connected vertices
        current = {idx_255}
        remaining = [i for i in range(n) if i != idx_255]
        random.shuffle(remaining)
        
        for i in remaining:
            if time.time() > deadline - 10.0:
                break
            if all(math.gcd(candidates[i], candidates[j]) > 1 for j in current):
                current.add(i)
        
        # Local search: try to add vertices
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            for i in range(n):
                if i in current:
                    continue
                if all(math.gcd(candidates[i], candidates[j]) > 1 for j in current):
                    current.add(i)
                    improved = True
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = {candidates[i] for i in current}
    
    # Convert to list
    result = list(best_set)
    # Ensure 255 is in the result
    if 255 not in result:
        result.append(255)
    
    return result
