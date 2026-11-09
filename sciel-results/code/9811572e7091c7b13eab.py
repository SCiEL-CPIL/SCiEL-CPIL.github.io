# EVOLVE-BLOCK-START
import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    p1, p2 = 5, 41
    
    # Quadratic residues (nonzero squares)
    qr1 = set((x*x) % p1 for x in range(1, p1))
    qr2 = set((x*x) % p2 for x in range(1, p2))
    
    # Build mapping: residue mod 205 -> (r mod 5, s mod 41)
    # CRT: x = a*41*(41^{-1} mod 5) + b*5*(5^{-1} mod 41) mod 205
    # 41 mod 5 = 1, so 41^{-1} mod 5 = 1
    # 5 mod 41 = 5, 5*33=165=4*41+1, so 5^{-1} mod 41 = 33
    # x = a*41 + b*165 mod 205
    inv_41_mod_5 = 1  # 41 ≡ 1 mod 5
    inv_5_mod_41 = 33  # 5*33 = 165 = 4*41+1
    
    # For each residue x in 0..204, compute (x mod 5, x mod 41)
    cell_of = [None] * m
    for x in range(m):
        r = x % p1
        s = x % p2
        cell_of[x] = (r, s)
    
    # For each cell (r, s), list of residues
    cell_to_residues = {}
    for x in range(m):
        c = cell_of[x]
        if c not in cell_to_residues:
            cell_to_residues[c] = []
        cell_to_residues[c].append(x)
    
    # All cells
    all_cells = list(cell_to_residues.keys())
    n_cells = len(all_cells)
    
    # Cell compatibility: cells (r1,s1) and (r2,s2) are compatible iff
    # (r1-r2)%5 not in qr1 OR (s1-s2)%41 not in qr2
    cell_compat = [[True]*n_cells for _ in range(n_cells)]
    for i in range(n_cells):
        r1, s1 = all_cells[i]
        for j in range(n_cells):
            if i == j:
                continue
            r2, s2 = all_cells[j]
            dr = (r1 - r2) % p1
            ds = (s1 - s2) % p2
            if dr in qr1 and ds in qr2:
                cell_compat[i][j] = False
    
    # For each pair of residues, check compatibility
    # res_compat[x][y] = True if x and y can coexist
    res_compat = [[True]*m for _ in range(m)]
    for x in range(m):
        for y in range(m):
            if x == y:
                continue
            d = (x - y) % m
            if d in qr1 and d in qr2:
                res_compat[x][y] = False
    
    # Precompute for each residue x, which cells are compatible with x
    # Actually, let's work at the cell level for B&B, then refine.
    
    # Degree of each cell (number of incompatible cells)
    cell_deg = [0] * n_cells
    for i in range(n_cells):
        for j in range(n_cells):
            if i != j and not cell_compat[i][j]:
                cell_deg[i] += 1
    
    # Order cells by degree descending
    cell_order = sorted(range(n_cells), key=lambda i: cell_deg[i], reverse=True)
    
    # Greedy coloring bound for the cell graph (independent set in cell graph)
    # We want max independent set in the cell compatibility graph.
    # Greedy coloring of the conflict graph (where edges = incompatible) gives upper bound.
    
    def greedy_color_bound_cell(available_mask):
        """Upper bound on max independent set in cell graph restricted to available cells."""
        if available_mask == 0:
            return 0
        
        colors = [-1] * n_cells
        num_colors = 0
        
        for ci in cell_order:
            if not (available_mask & (1 << ci)):
                continue
            
            # Find forbidden colors: colors of incompatible cells that are available
            forbidden = set()
            for cj in range(n_cells):
                if cj == ci:
                    continue
                if not (available_mask & (1 << cj)):
                    continue
                if not cell_compat[ci][cj]:
                    if colors[cj] != -1:
                        forbidden.add(colors[cj])
            
            c = 0
            while c in forbidden:
                c += 1
            colors[ci] = c
            if c >= num_colors:
                num_colors = c + 1
        
        return num_colors
    
    # B&B over cells to find max independent set of cells
    best_cell_set = [None]
    best_cell_size = [0]
    
    start_time = time.time()
    deadline = start_time + 570.0
    
    def bnb_cells(available_mask, chosen_cells, chosen_mask):
        if time.time() > deadline:
            return
        
        if available_mask == 0:
            if len(chosen_cells) > best_cell_size[0]:
                best_cell_size[0] = len(chosen_cells)
                best_cell_set[0] = list(chosen_cells)
            return
        
        bound = greedy_color_bound_cell(available_mask)
        if len(chosen_cells) + bound <= best_cell_size[0]:
            return
        
        # Pick next cell to branch on
        ci = -1
        for c in cell_order:
            if available_mask & (1 << c):
                ci = c
                break
        if ci == -1:
            return
        
        # Include ci
        new_avail = available_mask & ~(1 << ci)
        # Remove incompatible cells
        for cj in range(n_cells):
            if not cell_compat[ci][cj]:
                new_avail &= ~(1 << cj)
        chosen_cells.append(ci)
        chosen_mask |= (1 << ci)
        bnb_cells(new_avail, chosen_cells, chosen_mask)
        chosen_cells.pop()
        chosen_mask &= ~(1 << ci)
        
        # Exclude ci
        new_avail2 = available_mask & ~(1 << ci)
        bnb_cells(new_avail2, chosen_cells, chosen_mask)
    
    # Initial greedy cell solution
    greedy_cells = []
    avail = (1 << n_cells) - 1
    while avail:
        # Pick cell with min degree in remaining
        best_ci = -1
        best_deg = n_cells + 1
        for c in range(n_cells):
            if avail & (1 << c):
                deg = 0
                for cj in range(n_cells):
                    if cj != c and avail & (1 << cj) and not cell_compat[c][cj]:
                        deg += 1
                if deg < best_deg:
                    best_deg = deg
                    best_ci = c
        if best_ci == -1:
            break
        greedy_cells.append(best_ci)
        avail &= ~(1 << best_ci)
        for cj in range(n_cells):
            if not cell_compat[best_ci][cj]:
                avail &= ~(1 << cj)
    
    if len(greedy_cells) > best_cell_size[0]:
        best_cell_size[0] = len(greedy_cells)
        best_cell_set[0] = list(greedy_cells)
    
    # Run B&B
    bnb_cells((1 << n_cells) - 1, [], 0)
    
    # Now we have best_cell_set[0] - a set of compatible cells
    # For each cell, we need to pick one residue from cell_to_residues[cell]
    # such that all picked residues are pairwise compatible.
    
    selected_cells = best_cell_set[0] if best_cell_set[0] else greedy_cells
    
    # For each selected cell, try all residues and pick the one that works best
    # Use backtracking: for each cell in order, try each residue, check compatibility with already chosen
    
    # Order cells by number of residues (fewer first for pruning)
    cell_order_for_refine = sorted(selected_cells, key=lambda ci: len(cell_to_residues[all_cells[ci]]))
    
    best_residue_set = [None]
    best_residue_size = [0]
    
    def refine_bnb(idx, chosen_residues, chosen_mask):
        if time.time() > deadline:
            return
        
        if idx == len(cell_order_for_refine):
            if len(chosen_residues) > best_residue_size[0]:
                best_residue_size[0] = len(chosen_residues)
                best_residue_set[0] = list(chosen_residues)
            return
        
        ci = cell_order_for_refine[idx]
        cell = all_cells[ci]
        residues = cell_to_residues[cell]
        
        # Try each residue in this cell
        for x in residues:
            # Check compatibility with all chosen residues
            compatible = True
            for y in chosen_residues:
                if not res_compat[x][y]:
                    compatible = False
                    break
            if not compatible:
                continue
            
            chosen_residues.append(x)
            refine_bnb(idx + 1, chosen_residues, chosen_mask | (1 << x))
            chosen_residues.pop()
        
        # Also try skipping this cell entirely
        refine_bnb(idx + 1, chosen_residues, chosen_mask)
    
    # Initial greedy refinement
    greedy_residues = []
    for ci in cell_order_for_refine:
        cell = all_cells[ci]
        residues = cell_to_residues[cell]
        best_x = -1
        for x in residues:
            compatible = True
            for y in greedy_residues:
                if not res_compat[x][y]:
                    compatible = False
                    break
            if compatible:
                best_x = x
                break
        if best_x != -1:
            greedy_residues.append(best_x)
    
    if len(greedy_residues) > best_residue_size[0]:
        best_residue_size[0] = len(greedy_residues)
        best_residue_set[0] = list(greedy_residues)
    
    # Run refinement B&B
    refine_bnb(0, [], 0)
    
    final_set = best_residue_set[0] if best_residue_set[0] else greedy_residues
    
    # Verify and return
    result = sorted(final_set)
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
