import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # 1. Precompute conflict graph
    # Vertices are 2..128. Map value to index 0..126.
    # Edge (u, v) exists if including both u and v in the sequence causes a product collision.
    # Since the sequence is strictly increasing, u and v will be adjacent in the subsequence 
    # only if no other selected vertices lie between them. However, a simpler necessary condition 
    # for conflict is:
    # a) u == v (not possible in set)
    # b) u * v == u (impossible for v>1)
    # c) u * v == v (impossible for u>1)
    # d) u * v == w for some w in the set? No, that's 3 elements.
    # 
    # Actually, the "conflict" for MIS on subsequence selection is not just pairwise.
    # If we select u and v, they might not be adjacent.
    # However, if we view this as finding a subset S of {2..128} such that when sorted, 
    # all consecutive block products are unique, the pairwise condition "u*v is not equal to any other single element or pair product" is not sufficient 
    # because u*v could collide with x*y where x,y are also in S but not adjacent to u,v.
    #
    # Wait, the task asks for a sequence. The standard greedy/DFS approach checks local validity.
    # The MIS approach on a "conflict graph" where edge (u,v) means "u and v cannot both be in the sequence" is only valid if the conflict property is pairwise.
    # Is it?
    # If u and v are in the sequence, do they necessarily cause a collision if u*v == w (some other element)?
    # If u < w < v, and we pick u, w, v.
    # Blocks: u, w, v, u*w, w*v, u*w*v.
    # If u*w == v, then block [u,w] product is v, which is same as block [v]. Collision.
    # So if u*w = v, we cannot have u, w, v.
    # This is a 3-way conflict, not pairwise.
    #
    # Therefore, a simple pairwise conflict graph MIS is NOT sufficient to guarantee global validity 
    # unless we define the graph such that edges represent "cannot be adjacent" or similar.
    # But "cannot be adjacent" doesn't prevent u, w, v if u, w are not adjacent (e.g., u, x, w, v).
    #
    # Let's re-evaluate the "MIS" directive.
    # The directive says: "edge connects two numbers if including both in a valid sequence causes any consecutive block product collision".
    # This phrasing implies that if an edge exists, they cannot both be in the sequence.
    # This is only true if the collision is caused *directly* by their presence regardless of other elements.
    # As shown above, u*w=v is a collision. Does u and v conflict? No, u and v can be in the sequence if w is not.
    # Does u and w conflict? If we pick u and w, and w is the next element, then product u*w is generated. 
    # If v is also in the sequence later, and u*w == v, then we have a collision between block [u,w] and block [v].
    # So u and w are "conflicted" with v.
    #
    # This suggests that the problem is NOT a simple MIS on a static conflict graph of pairs.
    # However, we can still use a branch-and-bound strategy that is more efficient than random greedy.
    #
    # Alternative Interpretation: 
    # Maybe we can use the pairwise constraint "u*v is not in the set of candidates" as a hard constraint for adjacency?
    # If we enforce that for any adjacent pair (u, v) in the final sequence, u*v is NOT in the final sequence, 
    # does that guarantee all block products are unique?
    # No. u*v could equal x*y for non-adjacent x,y.
    #
    # Given the complexity, a pure MIS on a pairwise graph is likely incorrect for this specific "unique block products" constraint 
    # unless the graph is constructed very carefully to represent "cannot be in the same valid sequence".
    # But "cannot be in the same valid sequence" is a global property, not pairwise.
    #
    # Let's look at the inspirations. They use DFS and Greedy.
    # The current program uses Randomized Greedy.
    # The directive asks for MIS. If MIS is technically inapplicable or hard to define correctly for global constraints, 
    # I should implement a more structured search (like Backtracking with Pruning) which is the "MIS-like" approach 
    # (branch and bound) but tailored to the sequential constraint.
    #
    # Strategy: Backtracking Search with Pruning.
    # We try to build the sequence. At each step, we try to add the smallest possible valid number, 
    # but we also allow "skipping" numbers to explore different branches.
    # To make it efficient, we use a heuristic: 
    # 1. Estimate the maximum possible length from the current state (e.g., count of remaining numbers that are "compatible").
    # 2. If current_len + max_possible <= best_len, prune.
    #
    # This is effectively a Depth-First Branch and Bound.
    #
    # To improve on the current random greedy:
    # 1. Use deterministic ordering (smallest first) but with backtracking.
    # 2. Use a strong pruning bound.
    
    N = 128
    candidates = list(range(2, N + 1))
    
    best_seq = []
    best_len = 0
    
    # State variables for DFS
    current_seq = []
    current_prods = set()
    
    # To speed up, we can precompute some factors or just use the product check.
    # The product check is O(L) where L is length.
    
    def dfs(start_idx):
        nonlocal best_seq, best_len
        
        # Pruning: 
        # The maximum number of elements we can add is the number of candidates left.
        # A tighter bound: 
        # Each new element x must not collide with existing products.
        # Rough bound: len(candidates) - start_idx
        if len(current_seq) + (len(candidates) - start_idx) <= best_len:
            return
            
        # If we've checked all candidates
        if start_idx >= len(candidates):
            if len(current_seq) > best_len:
                best_len = len(current_seq)
                best_seq = current_seq[:]
            return
            
        # Time check
        if time.time() > deadline:
            return
            
        x = candidates[start_idx]
        
        # Option 1: Skip x
        dfs(start_idx + 1)
        
        # Option 2: Include x (if valid)
        # Check if x can be added
        # New suffix products: x, x*last, x*last*second_last, ...
        # We need to check if any of these are in current_prods.
        
        # Optimization: 
        # If x is in current_prods, it's invalid.
        if x not in current_prods:
            # Generate new suffix products
            new_prods = [x]
            prod = x
            valid = True
            # Iterate backwards through current_seq to build suffix products
            # current_seq is [a1, a2, ..., ak]
            # Suffixes of (current_seq + [x]) are:
            # x
            # x * ak
            # x * ak * a_{k-1}
            # ...
            # x * ... * a1
            
            for i in range(len(current_seq) - 1, -1, -1):
                prod *= current_seq[i]
                if prod in current_prods:
                    valid = False
                    break
                new_prods.append(prod)
            
            if valid:
                # Add x
                current_seq.append(x)
                current_prods.update(new_prods)
                
                dfs(start_idx + 1)
                
                # Backtrack
                current_seq.pop()
                for p in new_prods:
                    current_prods.discard(p)
        
        # Additional Pruning Heuristic:
        # If we skipped x, maybe we should also check if including x was the "only" way to get a long sequence?
        # The standard DFS explores all possibilities.
        # To guide the search, we can order the choices. 
        # Currently we try Skip then Include. 
        # It is often better to try Include first to find a good solution quickly (for pruning).
        # Let's swap: Try Include first, then Skip.
        
    # Let's re-implement DFS to try Include first.
    
    def dfs_include_first(start_idx):
        nonlocal best_seq, best_len
        
        if len(current_seq) + (len(candidates) - start_idx) <= best_len:
            return
            
        if start_idx >= len(candidates):
            if len(current_seq) > best_len:
                best_len = len(current_seq)
                best_seq = current_seq[:]
            return
            
        if time.time() > deadline:
            return
            
        x = candidates[start_idx]
        
        # Try Include x
        if x not in current_prods:
            new_prods = [x]
            prod = x
            valid = True
            for i in range(len(current_seq) - 1, -1, -1):
                prod *= current_seq[i]
                if prod in current_prods:
                    valid = False
                    break
                new_prods.append(prod)
            
            if valid:
                current_seq.append(x)
                current_prods.update(new_prods)
                
                dfs_include_first(start_idx + 1)
                
                current_seq.pop()
                for p in new_prods:
                    current_prods.discard(p)
        
        # Try Skip x
        dfs_include_first(start_idx + 1)
        
    # Run the DFS
    # To avoid hitting recursion limits or stack depth issues (though depth is only 127), 
    # Python's default recursion limit is 1000, which is fine.
    
    # We can also add a "randomized restart" or "different order" if time permits, 
    # but DFS with include-first is a very strong heuristic for maximizing length 
    # because it tends to pack in as many numbers as possible early on, 
    # and the pruning bound (remaining count) is effective.
    
    # However, pure DFS might get stuck in a local optimum if the "include first" strategy 
    # leads to a dead end with a shorter sequence than a path that skipped an early number.
    # The backtracking handles this, but the search space is 2^127.
    # The pruning is key. 
    # The bound `len(current_seq) + (len(candidates) - start_idx) <= best_len` is weak initially.
    # A stronger bound would be: "How many of the remaining numbers are compatible with the current suffix?"
    # But that's expensive to compute.
    
    # Let's run the DFS.
    dfs_include_first(0)
    
    # If the DFS didn't find a very good solution quickly, or if we have time left,
    # we can try some randomized greedy runs to see if we can beat the DFS result.
    # The DFS result is likely optimal or near-optimal if it completes.
    # But it might time out.
    
    if time.time() < deadline - 5.0:
        # Add a few randomized greedy runs to explore different "shapes" of the sequence
        random.seed(42)
        def run_greedy(p_add_base):
            seq = []
            prods = set()
            suffix_prods = []
            
            for x in range(2, 129):
                if random.random() < p_add_base:
                    new_sp = [x]
                    valid = True
                    if x in prods:
                        valid = False
                    else:
                        for p in suffix_prods:
                            np = x * p
                            if np in prods:
                                valid = False
                                break
                            new_sp.append(np)
                    
                    if valid:
                        seq.append(x)
                        prods.update(new_sp)
                        suffix_prods = new_sp
            return seq
        
        # Try a few different probabilities
        for p in [0.8, 0.9, 0.7, 0.85]:
            if time.time() > deadline - 5.0:
                break
            seq = run_greedy(p)
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq

    if not best_seq:
        # Fallback: simple greedy
        seq = []
        prods = set()
        suffix_prods = []
        for x in range(2, 129):
            new_sp = [x]
            valid = True
            if x in prods:
                valid = False
            else:
                for p in suffix_prods:
                    np = x * p
                    if np in prods:
                        valid = False
                        break
                    new_sp.append(np)
            if valid:
                seq.append(x)
                prods.update(new_sp)
                suffix_prods = new_sp
        best_seq = seq

    return best_seq

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
