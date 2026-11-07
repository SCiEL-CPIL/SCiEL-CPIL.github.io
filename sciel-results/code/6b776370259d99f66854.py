"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def construct():
    # Deadline guard: 150s limit, aim to finish well before
    start_time = time.time()
    deadline = start_time + 140.0 

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        # Clique on first k vertices
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges with probability p
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)

    def solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        
        # Map edges to indices for ILP
        edge_index = {e: i for i, e in enumerate(edges)}
        m = len(edges)
        
        # Generate candidate cliques
        # A clique in a split graph is a subset of C (clique part) plus a subset of S (independent part)
        # such that the S vertices are all connected to all C vertices in the clique.
        
        # 1. Get all cliques within C (first k vertices)
        # Since C is a clique, any subset is a clique.
        # We only need to consider subsets of C that are "maximal" with respect to their S-extensions?
        # Actually, for set cover, we want large cliques.
        # A clique is defined by a subset $C' \subseteq C$ and $S' \subseteq S$ such that $S'$ is a common neighbor of $C'$.
        
        # Let's enumerate subsets of C. 
        # k is small (10, 20, 30). 
        # 2^30 is too large for enumeration. 
        # However, we can be smarter.
        
        # Observation: The optimal solution will likely use cliques that are "maximal" in some sense.
        # But let's look at the structure.
        # Any clique consists of $C_{sub}$ and $S_{sub}$.
        # $S_{sub}$ must be a subset of $N(C_{sub}) \cap S$.
        # To cover edges efficiently, we want large cliques.
        # 
        # Alternative approach for candidate generation:
        # 1. All edges within C: covered by cliques contained in C.
        # 2. Cross edges: covered by cliques containing at least one C vertex and at least one S vertex.
        
        # Let's generate candidates based on "stars" or maximal cliques.
        # In a split graph, maximal cliques are of the form $C' \cup N(C')$ where $C' \subseteq C$.
        # Wait, $N(C')$ in S.
        
        # Since k <= 30, enumerating all subsets of C is $2^{30} \approx 10^9$, which is too slow.
        # We need a heuristic for candidate generation or a smarter ILP formulation.
        
        # Let's try a different angle:
        # The edges within C form a complete graph $K_k$.
        # The edges between C and S are random.
        
        # If we pick a clique $Q = C_Q \cup S_Q$, it covers:
        # - All edges within $C_Q$
        # - All edges between $C_Q$ and $S_Q$
        
        # We can decompose the problem?
        # No, the ILP formulation is standard.
        
        # Let's limit the number of candidates to avoid explosion.
        # We can generate candidates greedily or via maximal cliques.
        
        # Maximal cliques in a split graph:
        # A clique is maximal if it cannot be extended.
        # If we pick $C' \subseteq C$, the maximal extension in S is $N(C') \cap S$.
        # So candidates are $\{ C' \cup (N(C') \cap S) \mid C' \subseteq C \}$.
        # Still $2^k$ candidates.
        
        # For k=30, this is too many.
        # However, many subsets $C'$ will yield the same $N(C') \cap S$ or be subsets of others.
        # Specifically, if $C_1 \subset C_2$, then $N(C_2) \subseteq N(C_1)$.
        # The clique $C_2 \cup N(C_2)$ is "larger" in C but potentially smaller in S than $C_1 \cup N(C_1)$.
        
        # Let's try to generate a smaller set of "representative" cliques.
        # We know that edges within C must be covered.
        # The edges within C are dense.
        
        # Let's try a heuristic for candidate generation:
        # 1. For each vertex $v \in S$, consider the clique formed by $N(v) \cup \{v\}$.
        #    $N(v) \subseteq C$. Let $C_v = N(v)$.
        #    Candidate: $C_v \cup \{v\}$.
        #    This covers all edges incident to $v$ from $C$, and edges within $C_v$.
        # 2. For subsets of C?
        
        # Let's refine the candidate set:
        # Candidates 1: For each $v \in S$, $Q_v = N(v) \cup \{v\}$. (Count: n-k)
        # Candidates 2: For each pair $u, w \in C$, $Q_{u,w} = \{u, w\} \cup N(\{u, w\}) \cap S$.
        #               This is expensive ($k^2 \cdot n$).
        # Candidates 3: The whole clique $C \cup N(C) \cap S$? No, $N(C)$ might be empty.
        
        # Let's stick to a manageable number of candidates.
        # The edges within C ($K_k$) are the bottleneck for "waste" if we only use star-like cliques.
        # A star clique $C_v \cup \{v\}$ covers edges in $K_{C_v}$.
        
        # Let's generate:
        # 1. All $Q_v = N(v) \cup \{v\}$ for $v \in S$.
        # 2. All $Q_{\{u\}} = \{u\} \cup N(u) \cap S$ for $u \in C$. (Same as above if we iterate S? No, $N(u)$ is in S. $Q_{\{u\}}$ is a star centered at u. It covers edges $(u, s)$ for $s \in N(u)$ and no edges within C other than none? Wait. A clique $\{u\} \cup S_u$ covers edges $(u, s)$. It does NOT cover edges $(u, w)$ for $w \in C, w \neq u$.
        #    So star cliques centered in C do NOT cover C-C edges.
        #    Star cliques centered in S ($Q_v = N(v) \cup \{v\}$) DO cover C-C edges within $N(v)$.
        
        # Therefore, to cover C-C edges, we MUST use cliques that contain at least 2 vertices from C.
        # The cliques $Q_v$ contain $N(v)$ vertices from C.
        # If $N(v)$ is large, $Q_v$ covers many C-C edges.
        
        # Let's add candidates that are just subsets of C?
        # If we use a clique $C' \subseteq C$, it covers edges within $C'$.
        # It does NOT cover cross edges.
        # So we need a mix.
        
        # Let's generate a set of candidate cliques:
        # 1. For each $v \in S$: $C_v = N(v) \cap C$. Candidate: $C_v \cup \{v\}$.
        # 2. For each $u \in C$: Candidate $\{u\} \cup (N(u) \cap S)$. (Covers cross edges, no C-C edges).
        # 3. For each pair $u, w \in C$: Candidate $\{u, w\} \cup (N(u) \cap N(w) \cap S)$.
        #    This covers edge $(u,w)$ and cross edges to common neighbors.
        #    Number of candidates: $k(k-1)/2$. For k=30, 435 candidates. Very manageable.
        # 4. For each triple $u, v, w \in C$? $k^3/6$. For k=30, 4060. Manageable.
        #    Maybe just pairs and singles (from S perspective) are enough?
        
        # Let's include:
        # - All $Q_v = N(v) \cup \{v\}$ for $v \in S$.
        # - All $Q_{\{u, w\}} = \{u, w\} \cup (N(u) \cap N(w) \cap S)$ for $u < w \in C$.
        # - All $Q_{\{u\}} = \{u\} \cup (N(u) \cap S)$ for $u \in C$. (Just in case, though less useful for C-C).
        
        # Let's build the incidence matrix.
        # Rows: Edges. Columns: Candidate Cliques.
        # $A_{e, q} = 1$ if $e \in q$, else 0.
        
        candidates = []
        
        # Helper to create a candidate clique
        def add_candidate(c_list):
            if len(c_list) < 2:
                return
            # Check if it's a valid clique (it should be by construction, but verify)
            # In split graph, any subset of C is clique. Any subset of S is independent.
            # Mixed set is clique iff S-part is subset of N(C-part).
            c_part = sorted([x for x in c_list if x < k])
            s_part = sorted([x for x in c_list if x >= k])
            # Verify S-part neighbors
            for s in s_part:
                for c in c_part:
                    if (min(c,s), max(c,s)) not in edge_index:
                        # Not a clique
                        return
            candidates.append(frozenset(c_list))
            
        # 1. S-centered cliques
        for v in range(k, n):
            nbs_c = [u for u in range(k) if (u, v) in edge_index]
            add_candidate(nbs_c + [v])
            
        # 2. C-pair centered cliques
        for u in range(k):
            for w in range(u+1, k):
                common_s = [s for s in range(k, n) if (u, s) in edge_index and (w, s) in edge_index]
                add_candidate([u, w] + common_s)
                
        # 3. C-single centered cliques (stars from C)
        for u in range(k):
            nbs_s = [s for s in range(k, n) if (u, s) in edge_index]
            add_candidate([u] + nbs_s)
            
        # Deduplicate candidates
        candidates = list(set(candidates))
        
        if not candidates:
            # Fallback: individual edges
            return [list(e) for e in edges]
            
        num_cands = len(candidates)
        
        # Build A matrix
        # A is m x num_cands
        # Using sparse matrix for efficiency
        from scipy.sparse import lil_matrix
        A = lil_matrix((m, num_cands), dtype=int)
        
        for j, cand in enumerate(candidates):
            cand_list = list(cand)
            # Check all pairs in cand_list
            for i in range(len(cand_list)):
                for l in range(i+1, len(cand_list)):
                    e = (min(cand_list[i], cand_list[l]), max(cand_list[i], cand_list[l]))
                    if e in edge_index:
                        A[edge_index[e], j] = 1
                        
        A = A.tocsr()
        
        # ILP: min sum(x) s.t. A x >= 1, x in {0,1}
        c = np.ones(num_cands)
        
        # Constraints: A x >= 1
        # scipy milp requires: lb <= A x <= ub
        # Here: 1 <= A x <= inf
        constraints = LinearConstraint(A, lb=1, ub=np.inf)
        
        # Bounds: 0 <= x <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: x is integer
        integrality = np.ones(num_cands)
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if res.status == 0: # Optimal
                selected = [i for i in range(num_cands) if res.x[i] > 0.5]
                result_cliques = [list(candidates[i]) for i in selected]
                return result_cliques
            else:
                # Fallback if ILP fails
                pass
                
        except Exception as e:
            pass
            
        # Fallback: Greedy Set Cover
        # Uncovered edges
        uncovered = set(edges)
        result_cliques = []
        
        # Sort candidates by size descending? Or by edges covered / cost?
        # Cost is 1 for all. So maximize edges covered.
        
        # Precompute edges for each candidate
        cand_edges = []
        for cand in candidates:
            cand_list = list(cand)
            es = set()
            for i in range(len(cand_list)):
                for l in range(i+1, len(cand_list)):
                    e = (min(cand_list[i], cand_list[l]), max(cand_list[i], cand_list[l]))
                    if e in edge_index:
                        es.add(e)
            cand_edges.append(es)
            
        # Greedy
        while uncovered:
            best_idx = -1
            best_count = 0
            for i, es in enumerate(cand_edges):
                # Count how many uncovered edges this candidate covers
                # Optimization: only check if candidate has any uncovered edges
                # Simple check:
                count = 0
                for e in es:
                    if e in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = i
                    
            if best_idx == -1 or best_count == 0:
                # Should not happen if candidates cover all edges
                # Fallback: pick an uncovered edge and make it a clique
                e = next(iter(uncovered))
                result_cliques.append(list(e))
                uncovered.remove(e)
            else:
                result_cliques.append(list(candidates[best_idx]))
                # Remove covered edges
                for e in cand_edges[best_idx]:
                    uncovered.discard(e)
                    
        return result_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline - 10:
            # Not enough time, use trivial fallback for remaining
            # But we need to return valid result.
            # We'll just return edge cliques for remaining specs
            pass
            
        try:
            cliques = solve_split(*spec)
            # Validate types
            validated_cliques = []
            for c in cliques:
                if len(c) < 2:
                    continue
                # Ensure ints
                vc = [int(x) for x in c]
                # Ensure distinct
                if len(set(vc)) != len(vc):
                    continue
                validated_cliques.append(vc)
            results.append(validated_cliques)
        except Exception as e:
            # Fallback: individual edges
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            
    # Final type check and return
    final_result = []
    for cl_list in results:
        final_list = []
        for c in cl_list:
            final_list.append(tuple(c)) # Tuple or List? Contract says list or tuple. Let's use list.
            # Actually, contract says "Each clique is a list of at least two distinct int vertices".
            # So convert to list.
            final_list[-1] = list(final_list[-1])
        final_result.append(final_list)
        
    return final_result

# EVOLVE-BLOCK-END
