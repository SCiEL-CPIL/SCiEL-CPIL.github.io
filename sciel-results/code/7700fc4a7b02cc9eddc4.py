import time
import itertools

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Strategy:
    # 1. Use a known linear triple system construction on 16 points.
    #    The affine geometry AG(2, 4) has 16 points.
    #    Lines in AG(2, 4) have 4 points. A triple system requires triples of size 3.
    #    However, we can construct a Steiner Triple System packing or a specific design.
    
    # A better approach for 16 points:
    # The maximum number of triples in a linear triple system (2-(16,3,1) packing) is bounded by 16*15/6 = 40.
    # Actually, a Steiner Triple System STS(16) does not exist because 16 is not 1 or 3 mod 6.
    # The maximum size of a partial STS(16) is 20 triples (since each point appears in at most (15)/2 = 7.5 -> 7 times? No, max degree is 5 for STS(15), so for 16 it's different).
    # Let's look for a specific structure with small independence number.
    
    # We want to minimize the independence number (alpha).
    # For a 2-(16,3,1) packing, the density is high.
    # Let's try to construct a system using the binary vector space F2^4.
    # Points are 0..15.
    # Consider triples {a, b, a+b} in F2^4? 
    # If we take all triples {a, b, a+b} where a,b != 0 and a!=b, we get 15*14/2 = 105 triples?
    # But we must ensure linearity: any two triples share at most 1 point.
    # In F2^4, {a, b, a+b} and {c, d, c+d} share at most 1 point IF they are not the same triple and don't share a pair.
    # Do they share a pair?
    # If they share 2 points, say {a,b}, then the third point is a+b. So the triple is unique to the pair {a,b}.
    # Thus, the set of all triples {a, b, a+b} in F2^4 (with a,b nonzero, a!=b) is a linear triple system?
    # Wait, the triple {a,b,a+b} contains 0 if a=b? No a!=b.
    # It contains 0 if a+b=0 => a=b, which is excluded.
    # So 0 is only in triples {a, b, a+b} where a+b != 0.
    # Actually, 0 is in triples {a, b, a+b} if a+b=0? No.
    # 0 is in the triple {a, b, a+b} if 0=a, 0=b, or 0=a+b.
    # If 0=a+b, then b=a. But we require distinct points in a triple? 
    # The problem says "triples [a,b,c]". Usually implies distinct.
    # If a,b are distinct non-zero, a+b is non-zero and distinct from a,b.
    # So 0 is NOT in any triple of the form {a,b,a+b} with a,b nonzero distinct?
    # Wait. If we include 0 as a point, we need triples containing 0.
    # The construction {a, b, a+b} in F2^4 for all a,b in F2^4 \ {0}, a!=b generates triples that do NOT contain 0.
    # This leaves 0 isolated or in triples of a different form.
    
    # Let's use a different standard construction for 16 points.
    # The complement of a 2-(16,3,1) design?
    # Or use the 5-dimensional binary vector space? No, 16 points is 2^4.
    
    # Let's try a greedy approach but with a better heuristic:
    # Maximize the number of triples while maintaining linearity.
    # Then check alpha.
    
    # However, the prompt asks for a DIFFERENT strategy.
    # The current program uses random shuffling and greedy packing.
    # Let's use a **Backtracking Search with Pruning** to find a linear triple system with a specific property, or use a **Known Design**.
    
    # Known Design: The "Pasch" configuration or similar?
    # Actually, let's look at the independence number.
    # If we can find a system where every 5-set contains a triple, alpha <= 4.
    # If we can find a system where every 4-set contains a triple, alpha <= 3.
    # Is alpha=3 possible?
    # If alpha=3, every 4 points contain a triple.
    # This implies the complement of the hypergraph is very sparse.
    
    # Let's try to construct a system based on the **Hadamard Matrix** or **Group Action**.
    
    # Alternative Strategy: **Simulated Annealing** or **Local Search** on the set of triples.
    # Start with a random linear triple system.
    # Move: Remove a triple, add a different triple that maintains linearity.
    # Objective: Minimize the independence number (or maximize the minimum number of triples hit by any k-set).
    
    # Given the time limit, a full local search might be risky if the neighborhood is large.
    # But 16 points is small. Total triples = 560.
    # Max linear triples ~ 40.
    # State space is manageable.
    
    import random
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute pairs for each triple for fast conflict checking
    triple_pairs = {t: set(itertools.combinations(t, 2)) for t in all_triples}
    
    # We will perform a local search.
    # State: A set of triples `S`.
    # Constraint: For any t1, t2 in S, len(set(t1) & set(t2)) <= 1.
    # This is equivalent to: No pair (i,j) is contained in more than one triple in S.
    
    # Let's start with a valid linear triple system.
    # We can generate one greedily, but we want a good starting point for optimization.
    
    def is_linear(triples):
        used_pairs = set()
        for t in triples:
            for p in triple_pairs[t]:
                if p in used_pairs:
                    return False
                used_pairs.add(p)
        return True
        
    def get_alpha(triples):
        # Compute independence number
        # Triple set for O(1) lookup
        t_set = set(triples)
        
        # Check if alpha < k
        # We can binary search or just check small k.
        # Max possible alpha is 16.
        # We want to minimize alpha.
        
        # Heuristic: If the system is dense, alpha is small.
        # Let's check for alpha = 3, 4, 5...
        
        # To check if alpha < k, we need to ensure every (k-1)-set contains a triple?
        # No, alpha < k means there is NO independent set of size k.
        # An independent set of size k is a subset of k points that contains NO triple entirely.
        # So we check all C(16, k) subsets.
        
        # For k=3: Check if any 3 points form a triple? No, independent set of size 3 means 3 points with NO triple among them.
        # Wait. If {a,b,c} is a triple, then {a,b,c} is NOT independent.
        # An independent set is a set of points I such that no triple is a subset of I.
        
        # So, to check if alpha < 3, we need to ensure that EVERY 3 points form a triple?
        # No. If every 3 points form a triple, then any set of 3 points contains a triple (itself).
        # So no independent set of size 3 exists. Alpha < 3 => Alpha <= 2.
        
        # To check if alpha < 4, we need to ensure that EVERY 4 points contains at least one triple.
        
        # Let's implement a function to compute alpha exactly for small 16 points.
        # We can use a branch and bound or just iterate subsets if k is small.
        
        # Since we want to MINIMIZE alpha, we look for systems where alpha is 3 or 4.
        
        # Let's just compute the maximum independent set size using a simple backtracking or DP?
        # With 16 points, we can represent the hypergraph as a bitset of forbidden triples.
        
        # Better: Iterate k from 3 to 16.
        # If there exists an independent set of size k, alpha >= k.
        # We want the smallest k such that NO independent set of size k exists?
        # No, alpha is the size of the LARGEST independent set.
        # So we want to find the max k such that an independent set of size k exists.
        
        # To minimize alpha, we want the max k to be small.
        
        def has_independent_set_of_size(k):
            # Check if there is a subset of size k with no triple inside.
            # We can iterate all C(16, k) subsets.
            # C(16, 3) = 560
            # C(16, 4) = 1820
            # C(16, 5) = 4368
            # C(16, 6) = 8008
            # This is fast enough.
            
            for subset in itertools.combinations(points, k):
                subset_set = set(subset)
                # Check if any triple is contained in subset_set
                # We can just check if any triple in our system is a subset of subset_set
                # Or, precompute for the current system?
                
                # Efficient check:
                # For each triple in the system, if it's a subset of the candidate independent set, it's not independent.
                # Actually, we want to know if the candidate set IS independent.
                # It is independent if NO triple from the system is a subset of it.
                
                is_ind = True
                for t in triples:
                    if t[0] in subset_set and t[1] in subset_set and t[2] in subset_set:
                        is_ind = False
                        break
                if is_ind:
                    return True
            return False
        
        # Find max k
        # Start from high k? Or low?
        # We want to minimize alpha.
        # Let's check k=3, 4, 5...
        # If has_independent_set_of_size(3) is False, alpha <= 2.
        # If True, alpha >= 3.
        
        # To find exact alpha:
        # Alpha is the largest k such that has_independent_set_of_size(k) is True.
        
        # We can check k=16 down to 3?
        # Or k=3 up to 16?
        # Since we want small alpha, we expect the answer to be small.
        # Let's check k=3, 4, 5, 6...
        
        # Actually, if has_independent_set_of_size(k) is True, it might be True for k-1 too.
        # We want the MAX k.
        # So we should check from 16 down to 3.
        # But 16 is easy: if the system is empty, alpha=16.
        
        # Let's optimize:
        # We know alpha >= 3 usually.
        
        # Let's just compute it.
        
        # Optimization: Precompute for the current `triples` list a set of triples.
        # And maybe a map from pair to triples? No, linearity ensures unique pair.
        
        # The loop above `for t in triples` is O(|S|). |S| <= 40.
        # C(16, 8) = 12870. 12870 * 40 = 514,800 ops. Fast.
        
        # Let's find the max k.
        max_k = 2 # At least 2 points can be independent (no triple of size 2)
        # Actually, any 2 points are independent unless there is a triple of size 2, which there isn't.
        # So alpha >= 2.
        
        # Check k=3
        if has_independent_set_of_size(3):
            max_k = 3
            # Check k=4
            if has_independent_set_of_size(4):
                max_k = 4
                # Check k=5
                if has_independent_set_of_size(5):
                    max_k = 5
                    # Check k=6
                    if has_independent_set_of_size(6):
                        max_k = 6
                        # ... and so on
                        # We can just loop up
                        for k in range(7, 17):
                            if has_independent_set_of_size(k):
                                max_k = k
                            else:
                                break # If no set of size k, no set of size k+1
        return max_k

    # Initialize with a greedy packing
    best_alpha = 16
    best_system = []
    
    # Generate a few random starting points
    for _ in range(5):
        random.shuffle(all_triples)
        current = []
        used_pairs = set()
        for t in all_triples:
            if all(p not in used_pairs for p in triple_pairs[t]):
                current.append(t)
                for p in triple_pairs[t]:
                    used_pairs.add(p)
        
        alpha = get_alpha(current)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = current[:]
            
        if best_alpha <= 3:
            break
            
    # Local Search
    # State: current list of triples
    # Move: 
    #   1. Pick a triple t to remove.
    #   2. Pick a new triple t_new to add, such that the system remains linear.
    #   3. If new_alpha < old_alpha, accept.
    #   4. If new_alpha == old_alpha, accept with some probability (Simulated Annealing) or just keep exploring?
    #   Since we want to minimize, we only accept improvements or equal (to move to a better basin).
    
    # To make it faster, we can restrict moves.
    # Removing a triple frees up 3 pairs.
    # We can only add a triple that uses ONLY those freed pairs? No, it can use any available pairs.
    # But to maintain linearity, t_new must not share a pair with ANY existing triple.
    # Since we removed t, the pairs of t are now available.
    # Other pairs are still used.
    # So t_new must be a triple such that all its pairs are NOT in `used_pairs_after_removal`.
    
    # Let's implement a simple hill climbing.
    
    current_system = best_system[:]
    current_alpha = best_alpha
    
    # Precompute conflicts?
    # We need to be able to check if a triple can be added.
    
    def can_add(triple, used_pairs):
        return all(p not in used_pairs for p in triple_pairs[triple])
    
    # Time budget
    while time.time() - start_time < deadline:
        # Pick a random triple to remove
        if not current_system:
            break
        idx = random.randint(0, len(current_system) - 1)
        t_remove = current_system[idx]
        
        # Remove it
        used_pairs = set()
        for t in current_system:
            for p in triple_pairs[t]:
                used_pairs.add(p)
        
        # Free up pairs of t_remove
        for p in triple_pairs[t_remove]:
            used_pairs.discard(p)
            
        # Find candidate triples to add
        # Candidates are triples that can be added with the current used_pairs.
        # To be efficient, we can just iterate all_triples and check can_add.
        # Or, we can look for triples that share at least one pair with t_remove?
        # Not necessarily. But adding a triple that is "far" might be less likely to improve alpha?
        # Let's just pick a random valid replacement.
        
        # To avoid O(N) scan every time, we could maintain a list of available triples?
        # But 560 triples is small.
        
        # Let's try a few random replacements
        found_improvement = False
        for _ in range(20): # Try 20 random candidates
            t_new = random.choice(all_triples)
            if can_add(t_new, used_pairs):
                # Apply move
                new_system = current_system[:]
                new_system[idx] = t_new
                
                # Check linearity is maintained (it is by construction)
                # Check alpha
                new_alpha = get_alpha(new_system)
                
                if new_alpha < current_alpha:
                    current_system = new_system
                    current_alpha = new_alpha
                    found_improvement = True
                    break
                elif new_alpha == current_alpha:
                    # Accept to explore?
                    # Let's accept to allow movement
                    current_system = new_system
                    # current_alpha stays same
                    # We don't break, we keep looking for improvement in this iteration?
                    # Or accept and restart loop?
                    # Let's accept and break to restart the selection process.
                    break
        
        if current_alpha <= 3:
            break
            
    # Return the best system found
    return [list(t) for t in current_system]

# Run construct to test
if __name__ == "__main__":
    res = construct()
    print(res)
