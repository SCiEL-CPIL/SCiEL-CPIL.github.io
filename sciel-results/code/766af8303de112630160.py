import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s kill)
    
    # Step 1: Select moduli - all even numbers from 2 to 128
    moduli = [m for m in range(2, 129) if m % 2 == 0]
    n = len(moduli)
    
    # Precompute gcds
    gcds = [[gcd(moduli[i], moduli[j]) for j in range(n)] for i in range(n)]
    
    # For each modulus i, precompute the set of residues r in [0, m_i) that are "blocked"
    # by each already-assigned modulus j.
    # Actually, let's think differently.
    
    # The conflict condition: (r_i, m_i) and (r_j, m_j) are DISJOINT iff
    # r_i % gcd(m_i, m_j) != r_j % gcd(m_i, m_j)
    
    # So for a fixed modulus i and residue r_i, the residues r_j that CONFLICT with it
    # are those where r_j % gcd(m_i, m_j) == r_i % gcd(m_i, m_j).
    
    # Let's build a conflict graph where:
    # Nodes: (i, r) for each modulus i and residue r in [0, m_i)
    # Edge between (i, r_i) and (j, r_j) if they CONFLICT (i.e., are NOT disjoint)
    # We want a maximum independent set = maximum set of pairwise disjoint classes
    
    # But the graph is huge: sum of m_i for i in range = 2+4+6+...+128 = 64*65 = 4160 nodes
    # That's manageable for bitset operations.
    
    # Actually, let me reconsider. The number of nodes is sum_{m even, 2<=m<=128} m = 4160.
    # A 4160-node graph with bitsets is feasible.
    
    # Let's index nodes: node_id = (i, r) where i is index in moduli, r is residue
    # Total nodes: total_nodes = sum(moduli)
    
    total_nodes = sum(moduli)
    
    # Map (i, r) to node index
    # offset[i] = sum of moduli[0..i-1]
    offset = [0] * n
    for i in range(1, n):
        offset[i] = offset[i-1] + moduli[i-1]
    
    # Build adjacency: for each node (i, r), which nodes (j, r_j) conflict?
    # Conflict: r % gcd(m_i, m_j) == r_j % gcd(m_i, m_j)
    # So for fixed i, r, j: the conflicting r_j values are those with r_j % g == r % g where g = gcd(m_i, m_j)
    # In range [0, m_j), there are m_j/g such values.
    
    # Instead of building full adjacency list (could be large), let's use bitsets.
    # Each node has a bitset of conflicting nodes.
    
    # But 4160 bits per node * 4160 nodes = ~2MB, that's fine.
    
    # Actually, let's think about this more carefully for efficiency.
    
    # Alternative approach: Since we need one residue per modulus (we're choosing exactly one residue for each modulus we include), 
    # this is more like a constraint satisfaction problem where we assign residues to moduli.
    
    # Let me use a different structural approach:
    # - We decide which moduli to include (subset of the 64 even moduli)
    # - For each included modulus, we assign a residue
    # - All pairs must be disjoint
    
    # The key insight: for moduli i and j with gcd g, the residues must satisfy r_i % g != r_j % g.
    # This means the "residue classes modulo g" must be different.
    
    # For even moduli, gcd(m_i, m_j) is always even (at least 2).
    # So for any pair, we need r_i % g != r_j % g.
    
    # Let's try a branch-and-bound where:
    # - We process moduli in some order
    # - At each step, we try each possible residue for the current modulus
    # - We use bitsets to quickly check conflicts with already-assigned moduli
    
    # Precompute for each pair (i, j) the gcd
    # For each modulus i and residue r, precompute the "signature" r % gcd(m_i, m_j) for each j
    
    # Actually, let's use a cleaner bitset-based approach:
    
    # For each modulus i, for each residue r in [0, m_i), create a bitset of size n
    # where bit j is set if residue r for modulus i is COMPATIBLE with SOME residue for modulus j
    # But that's not quite right either.
    
    # Let me try a different structural approach: Maximum Independent Set on a carefully constructed graph.
    
    # Graph construction:
    # - One node per (modulus_index, residue) pair
    # - Two nodes are connected by an edge if they are from DIFFERENT moduli AND they CONFLICT
    # - Nodes from the SAME modulus are always "connected" (we can only pick one per modulus)
    # - We want the maximum independent set, which gives us the maximum set of pairwise disjoint classes
    
    # But we also want to maximize the number of moduli used, not just the number of nodes.
    # Since we pick at most one node per modulus, the number of nodes in the independent set = number of moduli used.
    
    # So the maximum independent set size = maximum number of moduli we can use.
    
    # Let's build this graph with bitsets.
    
    # Node indexing: node (i, r) has index offset[i] + r
    # Total nodes: total_nodes = 4160
    
    # For each node, build a bitset of its neighbors (conflicting nodes from other moduli + all other nodes from same modulus)
    
    # This is 4160 * 4160 bits = ~2MB, feasible.
    
    # Let's use Python integers as bitsets (they support bitwise ops efficiently).
    
    # neighbors[node_id] = integer whose bit k is set if node k is a neighbor of node_id
    
    # But 4160 bits in an integer is fine.
    
    # Let's build the neighbor bitsets.
    
    # For speed, let's precompute for each pair (i, j) with i < j:
    # g = gcd(m_i, m_j)
    # For residue r_i in [0, m_i): the conflicting residues r_j in [0, m_j) are those with r_j % g == r_i % g
    # There are m_j/g such residues.
    
    # Building full adjacency might be slow. Let's estimate:
    # For each pair (i, j), for each r_i (m_i values), we mark m_j/g neighbors.
    # Total work: sum over pairs (i,j) of m_i * (m_j/g)
    # For even moduli, g >= 2, so m_j/g <= m_j/2
    # Sum over pairs: sum_{i<j} m_i * m_j/2 approximately
    # That's (1/2) * (sum m_i)^2 - (1/2) * sum m_i^2 / 2... this could be large.
    
    # Let me compute: sum m_i = 4160, sum m_i^2 = 4^2 + 8^2 + ... + 128^2 = 4*(1^2 + 2^2 + ... + 64^2) = 4 * 64*65*129/6 = 4 * 69640 = 278560
    # Sum over pairs m_i * m_j = (4160^2 - 278560)/2 = (17305600 - 278560)/2 = 17027040/2 = 8513520
    # So total neighbor markings: ~8513520 * (avg m_j/g) ... this is getting complicated.
    
    # Let me just try to build it and see if it's fast enough.
    
    # Actually, a better approach: for each node (i, r_i), compute its neighbor bitset directly.
    
    # neighbors = [0] * total_nodes
    
    # For each i, for each r_i in [0, m_i):
    #   node_id = offset[i] + r_i
    #   For each j != i:
    #     g = gcds[i][j]
    #     For each r_j in [0, m_j) where r_j % g == r_i % g:
    #       neighbor_id = offset[j] + r_j
    #       neighbors[node_id] |= (1 << neighbor_id)
    #   Also, all other nodes in the same modulus i are neighbors:
    #     for r_i2 in [0, m_i) where r_i2 != r_i:
    #       neighbors[node_id] |= (1 << (offset[i] + r_i2))
    
    # This is O(total_nodes * n * avg_m/g) which might be too slow.
    
    # Let me try a different approach: use the structure more directly.
    
    # Key observation: For even moduli, gcd(m_i, m_j) is always a power of 2 times some odd factor.
    # Actually, gcd of two even numbers is even, but not necessarily a power of 2.
    
    # Let me try a simpler structural approach:
    # Use branch-and-bound with good pruning, processing moduli in an order that maximizes constraints early.
    
    # Order moduli by descending number of prime factors or descending value to constrain early.
    
    # Actually, let's try: process moduli in descending order of value.
    # Larger moduli have more residues to choose from, but also more constraints.
    
    # Hmm, let me try yet another approach:
    # Since all moduli are even, gcd(m_i, m_j) >= 2 for all pairs.
    # The constraint is r_i % g != r_j % g where g = gcd(m_i, m_j).
    
    # For g = 2: r_i % 2 != r_j % 2, so one must be even and the other odd.
    # This is a strong constraint!
    
    # So for any pair with gcd = 2, the residues must have different parity.
    
    # Let's check which pairs have gcd = 2:
    # gcd(2k, 2l) = 2*gcd(k, l), so gcd = 2 iff gcd(k, l) = 1.
    
    # This gives us a graph where edges connect moduli with gcd = 2, and the constraint is parity.
    
    # This is a 2-coloring constraint! If we can 2-color the graph of gcd=2 pairs, we're good for those.
    
    # But we also have constraints from gcd > 2.
    
    # Let me try a practical approach: use the current backtracking but with a much better heuristic and bitset-based conflict checking.
    
    # For each modulus i and residue r, precompute a bitset of which already-assigned moduli would conflict.
    
    # Actually, let me just optimize the current approach with bitsets.
    
    # State: assignment[i] = residue for modulus i, or None if unassigned
    # For each modulus i, maintain a bitset of "forbidden residues" based on already-assigned moduli.
    
    # forbidden[i] = set of residues r in [0, m_i) that conflict with some already-assigned modulus.
    
    # When we assign modulus j with residue r_j, for each unassigned modulus i:
    #   g = gcd(m_i, m_j)
    #   The residues r_i that conflict are those with r_i % g == r_j % g
    #   Add these to forbidden[i]
    
    # This is efficient! Let's implement this.
    
    # Order moduli: let's try descending order (largest first) to constrain early.
    # Or maybe ascending? Let's try both.
    
    # Actually, let's try a smart order: moduli with fewer "options" first.
    # A modulus m has m possible residues. After constraints, the number of valid residues decreases.
    # Start with moduli that have the most constraints (largest gcd with others).
    
    # For simplicity, let's try descending order of modulus value.
    
    moduli_desc = sorted(moduli, reverse=True)
    # Map from original index to new index
    # Actually, let's just work with the sorted list directly.
    
    n_desc = len(moduli_desc)
    gcds_desc = [[gcd(moduli_desc[i], moduli_desc[j]) for j in range(n_desc)] for i in range(n_desc)]
    
    # forbidden[i] = bitmask of forbidden residues for moduli_desc[i]
    # Use Python int as bitmask
    forbidden = [0] * n_desc
    assignment = [None] * n_desc
    
    def is_valid_residue(i, r):
        # Check if residue r is not forbidden for modulus i
        return not (forbidden[i] >> r) & 1
    
    def add_constraints(i, r):
        # Add constraints from assigning moduli_desc[i] = r to all unassigned moduli
        for j in range(n_desc):
            if j != i and assignment[j] is None:
                g = gcds_desc[i][j]
                # Residues r_j in [0, m_j) with r_j % g == r % g are forbidden
                target = r % g
                for rj in range(target, moduli_desc[j], g):
                    forbidden[j] |= (1 << rj)
    
    def remove_constraints(i, r):
        # Remove constraints from moduli_desc[i] = r from all unassigned moduli
        for j in range(n_desc):
            if j != i and assignment[j] is None:
                g = gcds_desc[i][j]
                target = r % g
                for rj in range(target, moduli_desc[j], g):
                    forbidden[j] &= ~(1 << rj)
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        
        if idx == n_desc:
            return True
        
        # Try to find a valid residue for moduli_desc[idx]
        m = moduli_desc[idx]
        
        # Get valid residues
        valid_residues = []
        for r in range(m):
            if is_valid_residue(idx, r):
                valid_residues.append(r)
        
        if not valid_residues:
            return False
        
        # Try each valid residue
        for r in valid_residues:
            assignment[idx] = r
            add_constraints(idx, r)
            if backtrack(idx + 1):
                return True
            remove_constraints(idx, r)
            assignment[idx] = None
        
        return False
    
    success = backtrack(0)
    
    if success:
        result = [[assignment[i], moduli_desc[i]] for i in range(n_desc)]
        return result
    else:
        # Fallback: greedy
        result = []
        for m in moduli:
            found = False
            for r in range(m):
                conflict = False
                for r2, m2 in result:
                    if r % gcd(m, m2) == r2 % gcd(m, m2):
                        conflict = True
                        break
                if not conflict:
                    result.append([r, m])
                    found = True
                    break
            if not found:
                continue
        return result
