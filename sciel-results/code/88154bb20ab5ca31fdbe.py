import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    best_score = 0.0
    best_selection = []
    
    def find_valid_residues(m, selected):
        """Find all valid residues for modulus m given selected classes."""
        if not selected:
            return list(range(m))
        # Mark forbidden residues
        forbidden = [False] * m
        for (m_j, r_j) in selected:
            g = math.gcd(m, m_j)
            r_j_mod = r_j % g
            for r in range(r_j_mod, m, g):
                forbidden[r] = True
        valid_rs = [r for r in range(m) if not forbidden[r]]
        return valid_rs
    
    def greedy_pass(randomize_r=False):
        selected = []
        for m in range(20, 201):
            if time.time() > deadline:
                break
            valid_rs = find_valid_residues(m, selected)
            if valid_rs:
                if randomize_r:
                    r = random.choice(valid_rs)
                else:
                    r = valid_rs[0]
                selected.append([m, r])
        
        score = sum(1.0 / m for m, r in selected)
        return selected, score
    
    def local_search(selected, max_iters=500):
        """Try to improve by swapping residues or adding/removing moduli."""
        selected = [list(x) for x in selected]
        current_score = sum(1.0 / m for m, r in selected)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 10:
            improved = False
            iters += 1
            # Try swapping residue for each selected modulus
            for i in range(len(selected)):
                m_i = selected[i][0]
                r_i = selected[i][1]
                others = [(selected[j][0], selected[j][1]) for j in range(len(selected)) if j != i]
                valid_rs = find_valid_residues(m_i, others)
                if len(valid_rs) > 1:
                    # Try each alternative residue
                    for r_new in valid_rs:
                        if r_new == r_i:
                            continue
                        # Check if this swap improves score (it doesn't change score since m is same)
                        # But it might enable adding more moduli later. Skip for now.
                        pass
            # Try adding a modulus that was skipped
            selected_ms = set(selected[i][0] for i in range(len(selected)))
            for m in range(20, 201):
                if m in selected_ms:
                    continue
                valid_rs = find_valid_residues(m, selected)
                if valid_rs:
                    # Try adding this modulus
                    selected.append([m, valid_rs[0]])
                    new_score = sum(1.0 / mm for mm, rr in selected)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        selected.pop()
            # Try removing a large modulus and replacing with two smaller ones?
            # Complex, skip for now.
        return selected, current_score
    
    # Run deterministic pass
    sel, score = greedy_pass(randomize_r=False)
    if score > best_score:
        best_score = score
        best_selection = sel
    
    # Run randomized passes with local search
    num_passes = 500
    for i in range(num_passes):
        if time.time() > deadline - 15.0:
            break
        sel, score = greedy_pass(randomize_r=True)
        if score > best_score:
            # Apply local search to improve
            sel_ls, score_ls = local_search(sel, max_iters=200)
            if score_ls > best_score:
                best_score = score_ls
                best_selection = sel_ls
            elif score > best_score:
                best_score = score
                best_selection = sel
    
    return best_selection
