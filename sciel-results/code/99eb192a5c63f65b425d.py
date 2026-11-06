import time
import random

def synthesize(matrix, depth_limit):
    start_time = time.time()
    # 90s wall clock limit. Use a generous internal deadline to allow more search.
    DEADLINE = start_time + 85.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Parse rows as bitmasks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Wire assignment: inputs 0..n-1, const-zero n, gates n+1...
    next_wire = n + 1
    gates = []
    # memo: bitmask -> wire_id
    memo = {}
    # Initialize with inputs
    for j in range(n):
        memo[1 << j] = j
    # Const zero
    memo[0] = n

    def elapsed():
        return time.time() - start_time

    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w

    def compute_rref_basis(row_masks):
        """
        Computes a basis for the row space using Gaussian Elimination.
        Returns a list of basis masks and a mapping from original row index to basis coordinates.
        """
        # We want to find a basis. We'll perform Gaussian elimination on the list of masks.
        # To keep the basis "small" in terms of popcount, we can use a variant of elimination
        # that prefers reducing popcount, but standard RREF is a good start.
        
        # Standard RREF logic for bitmasks
        # We iterate through columns from most significant to least? 
        # Actually, for XOR circuits, the specific pivot choice matters less than the span.
        # However, to minimize the "combination" cost later, we want the basis vectors to be sparse.
        
        basis = []
        # Sort by popcount descending might help find independent vectors faster?
        # Let's just iterate through rows.
        for mask in row_masks:
            if mask == 0:
                continue
            # Try to eliminate using existing basis
            temp = mask
            for b in basis:
                # If the highest set bit of temp is also the highest set bit of b,
                # XOR them to eliminate that bit.
                # This requires basis to be in reduced form (each pivot bit unique).
                # Let's maintain basis in a way that allows quick elimination.
                # Actually, standard Gaussian elimination:
                # Find if temp is in span of basis.
                pass
            
            # Simpler approach for basis:
            # Just collect independent vectors.
            # To ensure we can easily express original rows, we should keep the transformation.
            # But for synthesis, we just need the basis vectors themselves.
            # And we need to know which basis vectors sum to each original row.
            
            # Let's do a full RREF to get the coordinates.
            # We'll work with a list of [mask, coords] where coords is a bitmask of basis indices used.
            # This is getting complex. 
            # Alternative: 
            # 1. Find a basis B.
            # 2. For each original row R, find c such that R = sum(c_i * B_i).
            #    This is solving a linear system.
            
            # Let's just use a simple independent set finder.
            is_independent = True
            # Check if mask is in span of current basis
            # We can do this by trying to reduce mask using basis vectors.
            # If we reduce it to 0, it's dependent.
            # To do this efficiently, we should keep basis in a form where pivot bits are unique.
            
            # Let's rebuild basis in RREF form (pivot bits unique).
            # We'll store basis as list of (pivot_bit, mask).
            # If we add a new vector, we reduce it with existing pivots.
            
            # This function is a bit heavy to inline. Let's write a helper.
            pass

    def find_basis_and_coords(row_masks):
        """
        Returns:
        basis_masks: list of masks forming a basis
        coords: list of bitmasks, where coords[i] has bits set for basis vectors that sum to row_masks[i]
        """
        # basis[i] corresponds to the i-th basis vector
        # We maintain a list of (pivot_pos, mask) to allow fast reduction.
        # pivot_pos is the index of the highest set bit.
        
        basis_pivots = {} # pivot_pos -> (mask, index_in_basis_list)
        basis_list = []
        coords_list = []
        
        for r_mask in row_masks:
            # We want to represent r_mask as sum of basis vectors.
            # Start with r_mask and try to eliminate bits using basis.
            # However, standard RREF elimination modifies the rows.
            # We need the coordinates of the *original* rows in the basis.
            
            # Let's do standard Gaussian elimination on the matrix, tracking the row operations.
            # This is effectively finding the inverse of the transformation.
            
            # Simpler:
            # 1. Extract a basis from the rows.
            # 2. Solve for coordinates of each original row in that basis.
            
            # Step 1: Extract Basis
            # We'll use a greedy approach to pick independent rows.
            # To make coordinates easy to find, we'll keep the basis in "reduced" form relative to the order they were added?
            # No, let's just pick a basis, then solve the linear system for each row.
            
            pass

    # The above RREF logic is getting complicated to implement correctly and efficiently from scratch.
    # Let's use a simpler structural change:
    # **Greedy with Global Subexpression Sharing**
    # The current program already does this, but it processes rows one by one.
    # The structural change: **Sort rows by "difficulty" (popcount) and use a more aggressive search for splits.**
    
    # Actually, the "RREF" insight suggests that if we have many rows, we should identify the minimal set of "building blocks" (basis) and then combine them.
    # Let's implement a specific RREF-based synthesis:
    
    def get_rref_basis(row_masks):
        # Returns a list of masks that form a basis for the span of row_masks.
        # We use Gaussian elimination to find independent vectors.
        # We sort by popcount to keep basis vectors sparse (heuristic).
        sorted_masks = sorted(row_masks, key=lambda x: x.bit_count())
        
        basis = []
        for mask in sorted_masks:
            if mask == 0:
                continue
            # Check if mask is in span of basis
            # Reduce mask using basis
            # To do this efficiently, we assume basis is in a form where we can quickly reduce.
            # Let's just try to reduce mask by XORing with basis vectors if it reduces the popcount.
            # This is not standard RREF but a "sparse basis" heuristic.
            
            # Better: Standard RREF on the set of masks.
            # We'll iterate through columns from high to low.
            # This is O(n * |basis|) per mask.
            
            # Let's stick to a simpler "Independent Set" extraction.
            # If mask cannot be reduced to 0 by XORing with any combination of basis vectors, it's independent.
            # We can check this by seeing if mask is in the span.
            # Span check:
            # Try to reduce mask to 0 using basis.
            # If we maintain basis in "row echelon" form (pivot bits distinct), reduction is fast.
            
            # Let's maintain basis in row echelon form.
            # basis[i] has its highest set bit at position p_i, and p_i are distinct and decreasing.
            
            temp = mask
            # Try to eliminate highest bits using existing basis
            # We need to find if temp is in span of basis.
            # We can do this by iterating through basis in reverse order (lowest pivot to highest? No, highest to lowest).
            # Actually, if basis is in RREF, we can reduce temp by checking if its highest bit matches a pivot.
            
            # Let's just use a simple check:
            # If we add this mask to basis, does the rank increase?
            # We can just try to reduce temp using the current basis.
            # If it reduces to 0, it's dependent.
            
            # Efficient reduction:
            # We need basis to be in a form where we can quickly find if a bit is a pivot.
            # Let's keep a dict: pivot_bit -> basis_mask
            # And we ensure that when we add a new mask, we eliminate all existing pivots from it,
            # and then eliminate the new pivot from all existing basis masks. (Full RREF)
            
            # This is O(n * rank) per insertion.
            
            pass

    # Given the complexity of implementing full RREF with coordinate tracking in a short time,
    # let's try a different structural approach: **Two-Phase Synthesis**.
    # Phase 1: Synthesize a "core" set of gates that covers the most frequent sub-expressions.
    # Phase 2: Synthesize the remaining rows using the core.
    
    # However, the most robust improvement over the current greedy is to **allow the search to look deeper**.
    # The current `compute_mask` looks for `M` in memo such that `mask ^ M` is small.
    # We can improve this by considering **pairs of memo entries** that might combine to form a useful intermediate.
    
    # Let's try to implement the RREF approach properly, as it is the most promising structural change.
    
    def compute_basis_and_coords(row_masks):
        n_cols = max([mask.bit_length() for mask in row_masks]) if row_masks else 0
        # We will perform Gaussian Elimination to find a basis.
        # We track the coordinates of each original row in terms of the basis.
        # basis[i] is the i-th basis vector (mask).
        # coords[i] is a bitmask indicating which basis vectors sum to row_masks[i].
        
        # We'll use a list of "rows" in the augmented matrix [Mask | CoordMask]
        # But CoordMask can be large if rank is high.
        # Let's just find the basis, then solve for coords.
        
        # 1. Find Basis
        basis = []
        # To find a basis that is "good" for synthesis, we want vectors with low popcount.
        # We'll use a greedy approach: pick the row with lowest popcount that is independent.
        # This is not standard RREF, but might yield sparser basis vectors.
        
        # Let's stick to standard RREF for correctness and simplicity of coord extraction.
        # We'll create a list of [mask, coord_mask]
        # coord_mask starts as 0 for each row.
        # When we XOR row i with row j, coord_mask[i] ^= coord_mask[j].
        
        rows = []
        for i, mask in enumerate(row_masks):
            # coord_mask has bit i set initially? No, coord_mask represents the combination of ORIGINAL rows.
            # Wait, we want the combination of BASIS vectors.
            # Let's do this:
            # We perform elimination on the list of rows.
            # Each row is (mask, index_in_original).
            # We don't track coords during elimination if we just want the basis.
            # After finding the basis, we solve A x = b for each original row b.
            
            rows.append(mask)
            
        # Standard RREF on rows
        # We'll use a list of [mask, pivot_col]
        # Actually, let's just find a linearly independent subset.
        
        indep_rows = []
        # Sort by popcount to prefer sparse vectors in basis
        sorted_indices = sorted(range(len(row_masks)), key=lambda i: row_masks[i].bit_count())
        
        for idx in sorted_indices:
            mask = row_masks[idx]
            if mask == 0:
                continue
            # Check if mask is in span of indep_rows
            # Reduce mask using indep_rows
            # We need indep_rows to be in a form that allows fast reduction.
            # Let's keep them in a simple list and just try all combinations? No, too slow.
            # Let's keep them in "echelon" form.
            
            # If we keep indep_rows in echelon form (pivot bits distinct), reduction is fast.
            # Let's maintain a dict: pivot_bit -> mask
            # And ensure that when we add a mask, we eliminate all existing pivots from it.
            
            temp = mask
            # Eliminate existing pivots from temp
            # We need to know the pivots of indep_rows.
            # Let's store pivots separately.
            
            # This is getting messy. Let's use a library-like approach with numpy if available?
            # No, standard library only.
            
            # Let's use a simple O(R^2) span check for small R.
            # If rank is small (< 20), we can check span by trying all subsets? No.
            # We can check span by reducing temp using the basis vectors.
            # To reduce fast, we need basis in RREF.
            
            # Let's implement RREF insertion.
            # basis_masks: list of masks
            # basis_pivots: dict {pivot_bit: mask}
            
            # We'll do this in the main loop.
            pass

    # Okay, I will implement a simplified RREF that extracts a basis, and then for each original row,
    # I will determine which basis vectors it is a sum of by solving the linear system.
    # Since n is at most 16, and rank is at most 16, solving 16x16 systems is trivial.
    
    def get_basis_and_coords(row_masks, n):
        # 1. Find a basis using Gaussian Elimination
        # We'll use a list of masks.
        basis = []
        # We'll sort rows by popcount to get a sparse basis.
        sorted_rows = sorted(row_masks, key=lambda x: x.bit_count())
        
        for mask in sorted_rows:
            if mask == 0:
                continue
            # Check if mask is in span of basis
            # We reduce mask using basis.
            # To make this fast, we assume basis is in a form where we can quickly reduce.
            # Let's just try to reduce it.
            # If we maintain basis in "reduced" form (each pivot bit unique and not present in other basis vectors),
            # we can reduce mask by XORing with basis vectors if the highest bit of mask matches a pivot.
            
            # Let's maintain basis in RREF.
            # We'll store basis as a list of (pivot_bit, mask).
            
            # Actually, let's just use a simple independent set finder.
            # If the number of rows is small, we can just check independence.
            
            # Let's use a standard algorithm:
            # Iterate through rows.
            # For each row, try to eliminate its highest set bit using existing basis vectors.
            # If we can eliminate it to 0, it's dependent.
            # Otherwise, it's a new basis vector.
            
            # We need basis to be in a form that allows quick "eliminate highest bit" operation.
            # This means for each bit position, we should have at most one basis vector with that bit as its highest set bit.
            # And we should ensure that this bit is not present in any other basis vector (RREF).
            
            # Let's maintain:
            # pivot_to_basis: dict {bit_pos: mask}
            
            # This requires re-processing existing basis when adding a new one to maintain RREF.
            # This is O(R) per insertion.
            
            pass

    # Given the time constraints and the risk of bugs in a from-scratch RREF implementation,
    # I will use a **Heuristic RREF**:
    # 1. Sort rows by popcount.
    # 2. Greedily pick independent rows to form a basis.
    # 3. For each original row, find its coordinates in the basis by solving the linear system.
    
    def find_basis(row_masks):
        basis = []
        # We'll use a simple check: is mask in span of basis?
        # We'll reduce mask using basis.
        # To reduce, we need basis in a form that allows quick reduction.
        # Let's just use a list and do a simple reduction.
        
        # Optimization: Keep basis in a "reduced" state.
        # We'll store basis as a list of masks.
        # When adding a new mask, we reduce it by XORing with existing basis vectors if it reduces the popcount.
        # This doesn't guarantee RREF, but it keeps vectors sparse.
        
        for mask in sorted(row_masks, key=lambda x: x.bit_count()):
            if mask == 0:
                continue
            # Try to reduce mask using existing basis
            reduced = mask
            changed = True
            while changed:
                changed = False
                for b in basis:
                    if (reduced ^ b).bit_count() < reduced.bit_count():
                        reduced ^= b
                        changed = True
            
            if reduced == 0:
                continue # Dependent
            
            # Add reduced to basis
            # And also reduce existing basis vectors with the new one to keep them sparse?
            # This is getting expensive.
            
            # Let's just add it.
            basis.append(reduced)
            
            # Optional: Reduce existing basis with new one
            # new_basis = []
            # for b in basis:
            #    if b == reduced: continue
            #    if (b ^ reduced).bit_count() < b.bit_count():
            #        b ^= reduced
            #    new_basis.append(b)
            # basis = new_basis
            
        return basis

    def get_coords(row_mask, basis):
        """
        Finds coordinates of row_mask in terms of basis.
        Returns a bitmask of indices into basis.
        """
        # Solve sum(x_i * basis[i]) = row_mask
        # We can do this by reducing row_mask using basis.
        # If we reduce it to 0, the XORs we performed give the coordinates.
        
        coords = 0
        temp = row_mask
        # We need to know which basis vectors we used.
        # This is tricky if basis is not in RREF.
        
        # Let's use a standard Gaussian Elimination on the basis vectors to find the coordinates.
        # Since basis size is small, we can just try all combinations? No.
        
        # We'll perform elimination on temp using basis.
        # We'll track which basis vectors were used.
        
        # To do this efficiently, we need basis in a form where we can quickly eliminate bits.
        # Let's assume basis is in "echelon" form (pivot bits distinct).
        # If not, we can convert it.
        
        # Let's just use a simple iterative reduction.
        # If we can't guarantee RREF, we might fail to find coordinates.
        
        # Better approach:
        # Build a matrix from basis vectors and solve for temp.
        
        k = len(basis)
        if k == 0:
            return 0
        
        # We'll use Gaussian elimination on the basis vectors to find the representation.
        # This is equivalent to finding the row of the inverse of the basis matrix.
        
        # Let's just do a simple search if k is small?
        # No, k can be up to 16. 2^16 is 65536. Too slow for 16 rows.
        
        # Let's implement a proper solver.
        # We'll create an augmented matrix [basis | e_i] and reduce to RREF.
        # Then the last column gives the coordinates for each basis vector's identity.
        # Wait, we want to express temp as a combo of basis.
        # We can append temp as a column and reduce.
        
        # Actually, we can just reduce temp using the basis vectors if we have them in RREF.
        # Let's ensure basis is in RREF.
        
        # I will skip the RREF complexity and use the **current program's logic** but with a **better sorting and a slightly deeper search**.
        # The structural change will be: **Use a "Core" set of gates derived from the most frequent bits.**
        
        return 0

    # RE-EVALUATION:
    # The current program is actually quite good. The score 1.4293 is decent.
    # The "RREF" inspiration suggests that explicitly sharing sub-expressions across rows is key.
    # The current program *does* share sub-expressions via memo.
    # The difference is that the current program picks the *next* row to process greedily.
    # A structural change: **Process rows in an order that maximizes the reuse of existing gates.**
    # i.e., Instead of sorting by popcount, sort by "closeness" to existing memo entries.
    
    # Let's implement this:
    # 1. Initialize memo with inputs.
    # 2. Loop:
    #    a. For each unprocessed row, calculate a "cost" to synthesize it given current memo.
    #       Cost = min over all M in memo of (cost of mask ^ M).
    #       Approx cost of mask ^ M is popcount(mask ^ M) if not in memo.
    #    b. Pick the row with the lowest cost.
    #    c. Synthesize it using the best M.
    #    d. Add to memo.
    
    # This is a **Dynamic Ordering** strategy. It is structurally different from static sorting.
    
    def synthesize_dynamic(row_masks, n, memo, gates, next_wire):
        # Returns updated memo, gates, next_wire, and outputs
        outputs = [None] * len(row_masks)
        unprocessed = set(range(len(row_masks)))
        
        def get_cost(mask):
            if mask in memo:
                return 0
            min_cost = float('inf')
            # Find best M in memo
            # Optimization: only check a subset of memo if it's large
            for M in memo:
                rem = mask ^ M
                if rem in memo:
                    cost = 0
                else:
                    cost = rem.bit_count()
                if cost < min_cost:
                    min_cost = cost
            return min_cost

        def build_mask(mask):
            if mask in memo:
                return memo[mask]
            
            # Find best M
            best_M = 0
            best_rem = mask
            min_cost = float('inf')
            
            for M, w_m in memo.items():
                rem = mask ^ M
                if rem in memo:
                    cost = 0
                else:
                    cost = rem.bit_count()
                
                if cost < min_cost:
                    min_cost = cost
                    best_M = M
                    best_rem = rem
                    if cost == 0:
                        break
            
            if best_rem in memo:
                w_rem = memo[best_rem]
            else:
                # Build rem from scratch (tree)
                bits = [j for j in range(n) if (best_rem >> j) & 1]
                # Simple tree build
                if not bits:
                    w_rem = n
                elif len(bits) == 1:
                    w_rem = bits[0]
                else:
                    wires = bits[:]
                    while len(wires) > 1:
                        new_wires = []
                        for i in range(0, len(wires), 2):
                            if i + 1 < len(wires):
                                new_w = add_gate(wires[i], wires[i+1])
                                new_wires.append(new_w)
                            else:
                                new_wires.append(wires[i])
                        wires = new_wires
                    w_rem = wires[0]
            
            w_m = memo[best_M]
            w_new = add_gate(w_m, w_rem)
            memo[mask] = w_new
            return w_new

        while unprocessed:
            # Find row with min cost
            best_idx = None
            best_cost = float('inf')
            for idx in unprocessed:
                cost = get_cost(row_masks[idx])
                if cost < best_cost:
                    best_cost = cost
                    best_idx = idx
            
            if best_idx is None:
                break
            
            outputs[best_idx] = build_mask(row_masks[best_idx])
            unprocessed.remove(best_idx)
            
            if elapsed() > DEADLINE - 5.0:
                # Fast path for remaining
                for idx in unprocessed:
                    outputs[idx] = build_mask(row_masks[idx])
                break

        return outputs, memo, gates, next_wire

    # Run dynamic synthesis
    outputs, memo, gates, next_wire = synthesize_dynamic(row_masks, n, memo, gates, next_wire)

    # Fill any None
    for i in range(m):
        if outputs[i] is None:
            outputs[i] = n

    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
