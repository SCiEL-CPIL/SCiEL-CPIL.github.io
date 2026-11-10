"""Maximize pairwise-disjoint residue classes using CSP backtracking with MRV."""
# EVOLVE-BLOCK-START
import time
from math import gcd

MAX_MOD = 128
DEADLINE = 85.0  # Safety margin before 90s kill

# Precompute GCD matrix
GCD = [[0] * (MAX_MOD + 1) for _ in range(MAX_MOD + 1)]
for i in range(2, MAX_MOD + 1):
    for j in range(i, MAX_MOD + 1):
        g = gcd(i, j)
        GCD[i][j] = g
        GCD[j][i] = g

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE
    
    # We want to select a subset of moduli from [2, 128] and assign residues.
    # To maximize the count, we should prioritize moduli that are "compatible" with many others.
    # Moduli with larger GCDs with other moduli restrict residues more, but if we can assign them,
    # they "use up" fewer distinct residue classes? No, they restrict more.
    # Actually, moduli with GCD=1 (coprime) with many others are "flexible" but don't restrict much?
    # Wait, the condition is disjointness.
    # If gcd(m1, m2) = 1, then (r1 - r2) % 1 == 0 is ALWAYS true.
    # So if gcd(m1, m2) == 1, the classes are NEVER disjoint?
    # Let's re-read the criterion: "disjoint iff gcd(m,m') ∤ (r-r')"
    # If gcd = 1, 1 always divides (r-r'). So (r-r') % 1 == 0.
    # The condition for NOT disjoint is (r-r') % g == 0.
    # So if g=1, they are NEVER disjoint.
    # Therefore, we MUST only pick moduli such that gcd(m_i, m_j) > 1 for all pairs?
    # Wait. If g=1, the intersection is non-empty. So they are NOT disjoint.
    # Thus, any valid set of disjoint classes MUST consist of moduli that are pairwise non-coprime?
    # Let's check:
    # Classes A: r=0, m=2 (even numbers)
    # Class B: r=1, m=3 (numbers 1, 4, 7...)
    # gcd(2,3)=1.
    # Intersection: x = 2k, x = 3j + 1.
    # 2k = 3j + 1 => 2k - 3j = 1. Has solutions (e.g., k=2, j=1 -> 4=4).
    # So they intersect.
    # Correct. If gcd(m1, m2)=1, the classes intersect.
    # Therefore, all moduli in the set must have gcd > 1 with each other.
    # This means the graph of moduli where edges connect non-coprime pairs must contain a clique.
    # We need to find the maximum clique in the graph G=(V, E) where V={2..128} and (u,v) in E iff gcd(u,v)>1.
    # AND we must assign residues such that for each edge (u,v), (r_u - r_v) % gcd(u,v) != 0.
    
    # So Step 1: Find a large clique in the non-coprime graph.
    # Step 2: Assign residues to this clique to satisfy the constraints.
    
    # Since we need to maximize the number of classes, and the residue assignment is always possible 
    # for a clique? 
    # Let's check if a clique in the non-coprime graph always admits a valid residue assignment.
    # Constraints: For each pair (i,j) in clique, r_i != r_j (mod g_ij).
    # This is a system of linear congruences? No, it's inequality constraints modulo g_ij.
    # Since g_ij >= 2, we have at least 1 forbidden residue for each pair relation?
    # Actually, for a fixed i, the constraint on r_i is:
    # r_i not in { r_j + k * g_ij | k in Z } mod m_i?
    # No, the condition is (r_i - r_j) % g_ij != 0.
    # This means r_i mod g_ij != r_j mod g_ij.
    # So for each pair (i,j), r_i is forbidden from being congruent to r_j modulo g_ij.
    
    # This is a CSP on the residues.
    # Given a clique of moduli, can we always find residues?
    # Consider a clique of powers of 2: 2, 4, 8, 16...
    # gcd(2,4)=2. r2 != r4 (mod 2).
    # gcd(4,8)=4. r4 != r8 (mod 4).
    # gcd(2,8)=2. r2 != r8 (mod 2).
    # This seems solvable.
    # What about a clique of multiples of 6? 6, 12, 18, 24...
    # gcd(6,12)=6. r6 != r12 (mod 6).
    # Since r6 in [0,5] and r12 in [0,11], we can just pick r6=0, r12=1.
    # It seems highly likely that any clique in the non-coprime graph has a valid residue assignment.
    # Why? Because the constraints are "not equal mod g". Since g >= 2, there are at least m_i - (m_i/g) * 1? 
    # Actually, the number of forbidden residues for r_i given r_j is m_i / g_ij.
    # If we have many constraints, it might fail?
    # But we can use backtracking to find the residues.
    
    # So the strategy is:
    # 1. Build the graph of non-coprime moduli.
    # 2. Use a Maximum Clique algorithm (e.g., Bron-Kerbosch with pivot) to find a large clique.
    # 3. Use CSP backtracking to assign residues to the clique.
    # 4. If step 3 fails (unlikely but possible), try the next largest clique or backtrack in clique selection.
    
    # However, finding the EXACT maximum clique in a graph of 127 nodes is NP-hard.
    # But 127 is small enough for optimized Bron-Kerbosch?
    # Or we can use a greedy approach to find a "good" clique, then refine.
    # The current program does local search on residues, but doesn't explicitly optimize the modulus set via clique finding.
    # The "disjointness" requirement forces the moduli to be pairwise non-coprime.
    # So the set of moduli MUST be a clique in the non-coprime graph.
    
    # Let's implement a heuristic Maximum Clique finder.
    # Bron-Kerbosch with pivot.
    
    # Graph representation:
    # neighbors[v] = set of u such that gcd(u,v) > 1
    
    neighbors = [set() for _ in range(MAX_MOD + 1)]
    for u in range(2, MAX_MOD + 1):
        for v in range(u + 1, MAX_MOD + 1):
            if GCD[u][v] > 1:
                neighbors[u].add(v)
                neighbors[v].add(u)
                
    # We want to find a large clique.
    # We can use a greedy heuristic:
    # 1. Sort vertices by degree (descending).
    # 2. Pick a vertex, add to clique.
    # 3. Remove all non-neighbors.
    # 4. Repeat.
    # This is a "Greedy Coloring" style or "Greedy Clique".
    # Better: Use a simple backtracking for Maximum Clique with a time limit, or use a greedy approximation and then local search on the clique (swap vertices).
    
    # Given the small size (127 nodes), we can try to find the maximum clique using a recursive search with pruning.
    # But we also need to assign residues.
    
    # Let's combine:
    # Find a large clique C.
    # Try to assign residues to C.
    # If success, return.
    # If fail, or if we can find a larger clique, keep searching.
    
    # Since the residue assignment is likely always possible for a clique, the bottleneck is finding the max clique.
    # Let's use a heuristic to find a large clique quickly, then verify residues.
    
    # Heuristic:
    # 1. Start with empty clique.
    # 2. Candidate set = all nodes.
    # 3. While candidates exist:
    #    a. Pick node v in candidates with max degree within candidates.
    #    b. Add v to clique.
    #    c. Candidates = Candidates ∩ neighbors[v].
    # This is a standard greedy clique algorithm.
    
    # Let's improve it:
    # Try multiple random orderings or pivots.
    
    def greedy_clique(candidates, time_limit=0.5):
        """
        Greedy clique finding with a bit of randomness or multiple passes.
        Returns a list of moduli.
        """
        if not candidates:
            return []
        
        clique = []
        # We'll do a few greedy passes with different tie-breaking
        best_clique = []
        
        for _ in range(20): # 20 random-ish passes
            cands = set(candidates)
            curr_clique = []
            while cands:
                # Pick node with max degree in cands
                # To add variety, we can pick randomly from top 3 degrees
                degs = [(len(cands.intersection(neighbors[v])), v) for v in cands]
                degs.sort(reverse=True)
                
                # Pick from top few
                top_k = degs[:min(5, len(degs))]
                v = top_k[0][1] if len(top_k) == 1 else __import__('random').choice([x[1] for x in top_k])
                
                curr_clique.append(v)
                # Intersect with neighbors
                new_cands = set()
                for u in cands:
                    if u in neighbors[v]:
                        new_cands.add(u)
                cands = new_cands
            
            if len(curr_clique) > len(best_clique):
                best_clique = curr_clique
                if len(best_clique) > 40: # Heuristic upper bound?
                    break
            if time.time() > time_limit:
                break
                
        return best_clique

    # Initial candidate set is all moduli 2..128
    all_mods = set(range(2, MAX_MOD + 1))
    
    # Find a large clique
    clique = greedy_clique(all_mods, time_limit=5.0)
    
    # Now assign residues to this clique using CSP backtracking
    # Variables: indices in clique
    # Domains: residues 0..m-1
    # Constraints: for i<j, (r_i - r_j) % gcd(m_i, m_j) != 0
    
    def assign_residues(clique_mods):
        """
        Backtracking solver for residue assignment.
        Returns list of [residue, modulus] or None.
        """
        n = len(clique_mods)
        if n == 0:
            return []
        
        # Precompute constraints for each pair
        # constraints[i] = list of (j, g)
        constraints = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                g = GCD[clique_mods[i]][clique_mods[j]]
                constraints[i].append((j, g))
                constraints[j].append((i, g))
                
        # Order variables by MRV (Minimum Remaining Values)
        # Initially, domain size is m_i.
        # We want to pick the variable with smallest domain first?
        # Or most constrained?
        # Let's use a simple heuristic: sort by modulus (smaller modulus = smaller domain).
        # This is a good proxy for MRV initially.
        
        order = sorted(range(n), key=lambda i: clique_mods[i])
        
        assignment = [None] * n
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == n:
                return True
            
            i = order[idx]
            m = clique_mods[i]
            
            # Try residues 0..m-1
            # Optimization: We can check constraints against already assigned variables
            for r in range(m):
                valid = True
                for j, g in constraints[i]:
                    if assignment[j] is not None:
                        rj = assignment[j]
                        if (r - rj) % g == 0:
                            valid = False
                            break
                if valid:
                    assignment[i] = r
                    if backtrack(idx + 1):
                        return True
                    assignment[i] = None
                    
            return False

        if backtrack(0):
            result = []
            for i in range(n):
                result.append([assignment[i], clique_mods[i]])
            return result
        else:
            return None

    # Try to assign residues to the greedy clique
    result = assign_residues(clique)
    
    if result is not None:
        return result
        
    # If greedy clique failed residue assignment (unlikely), try to find a smaller clique or different clique
    # For robustness, let's try a few more clique variations if the first one fails
    # Or just return a trivial solution if time is running out
    
    if time.time() > deadline - 5:
        return [[0, 2]]
        
    # Fallback: Try to build a clique from powers of 2 (always works)
    # 2, 4, 8, 16, 32, 64, 128
    # gcds are powers of 2.
    # Residues:
    # 2: 0
    # 4: 1 (1%2 != 0%2)
    # 8: 2 (2%4 != 1%4, 2%2 != 0%2)
    # 16: 4
    # 32: 8
    # 64: 16
    # 128: 32
    # This is a valid clique of size 7.
    
    fallback_clique = [2, 4, 8, 16, 32, 64, 128]
    fallback_result = assign_residues(fallback_clique)
    if fallback_result:
        return fallback_result
        
    return [[0, 2]]
# EVOLVE-BLOCK-END
