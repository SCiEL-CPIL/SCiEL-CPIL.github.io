"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
import sys
from itertools import combinations

_START_TIME = None

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Safety margin: 85s out of 90s
    deadline = _START_TIME + 85.0
    
    moduli = list(range(2, 129))
    n = len(moduli)
    
    # Build adjacency list for the "compatible" graph
    # Edge (u, v) exists if gcd(moduli[u], moduli[v]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if _gcd(moduli[i], moduli[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
                
    # Find Maximum Clique using Bron-Kerbosch with pivoting
    # Since N=127 is small, this is fast.
    best_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
        
        # Choose pivot u from P union X to maximize |N(u) ∩ P|
        union_px = P.union(X)
        if not union_px:
            return
            
        # Simple pivot selection: pick one that maximizes intersections with P
        # For speed, we can just pick the first one or random. 
        # With N=127, even a simple heuristic is fine.
        pivot_candidates = list(union_px)
        # Sort by degree in P to pick a good pivot
        pivot_candidates.sort(key=lambda u: len(adj[u].intersection(P)), reverse=True)
        u = pivot_candidates[0]
        
        # P \ N(u) are the candidates to recurse on
        candidates = P.difference(adj[u])
        
        for v in list(candidates):
            new_R = R + [v]
            new_P = P.intersection(adj[v])
            new_X = X.intersection(adj[v])
            bron_kerbosch(new_R, new_P, new_X)
            P.remove(v)
            X.add(v)
            
    # Initialize
    P = set(range(n))
    X = set()
    R = []
    bron_kerbosch(R, P, X)
    
    # The best_clique contains indices of moduli that form a clique (pairwise non-coprime)
    # Now we need to assign residues to these moduli such that they are pairwise disjoint.
    # Condition: (r_i - r_j) % gcd(m_i, m_j) != 0
    
    selected_moduli = [moduli[idx] for idx in best_clique]
    
    # If the clique is small or empty, fallback to a simple valid set
    if not selected_moduli:
        return [[1, 2]]
        
    # Backtracking to assign residues
    # Sort selected moduli by size of gcd with others? 
    # Just use the order from the clique.
    # To speed up backtracking, sort by "constraint tightness" (higher gcds are tighter)
    # Actually, just assign in the order they appear.
    
    assignment = {}
    
    def assign_residues(idx):
        if idx == len(selected_moduli):
            return True
            
        m = selected_moduli[idx]
        # Try residues 0 to m-1
        # Heuristic: try residues that are "far" from existing assignments?
        # Just iterate 0..m-1
        for r in range(m):
            valid = True
            # Check against all previously assigned moduli
            for j in range(idx):
                m_j = selected_moduli[j]
                r_j = assignment[selected_moduli[j]]
                g = _gcd(m, m_j)
                if (r - r_j) % g == 0:
                    valid = False
                    break
            if valid:
                assignment[m] = r
                if assign_residues(idx + 1):
                    return True
                del assignment[m]
                
        return False
        
    # To make backtracking faster, we can reorder the moduli in the clique
    # Moduli with smaller m have fewer residue choices? No, m is the modulus.
    # Moduli with high gcds with many others are more constrained.
    # Let's compute a constraint score for each modulus in the clique
    scores = []
    for m in selected_moduli:
        score = 0
        for m2 in selected_moduli:
            if m != m2:
                score += _gcd(m, m2)
        scores.append((score, m))
        
    # Sort by score descending (most constrained first)
    scores.sort(reverse=True)
    ordered_moduli = [m for _, m in scores]
    
    # Re-run backtracking with ordered moduli
    assignment = {}
    def assign_residues_ordered(idx):
        if idx == len(ordered_moduli):
            return True
            
        m = ordered_moduli[idx]
        # Randomize residue order to avoid getting stuck in a specific bad pattern
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            valid = True
            for j in range(idx):
                m_j = ordered_moduli[j]
                r_j = assignment[m_j]
                g = _gcd(m, m_j)
                if (r - r_j) % g == 0:
                    valid = False
                    break
            if valid:
                assignment[m] = r
                if assign_residues_ordered(idx + 1):
                    return True
                del assignment[m]
                
        return False
        
    # Try a few times with different random seeds if it fails
    success = False
    for attempt in range(100):
        if time.time() > deadline:
            break
        assignment = {}
        if assign_residues_ordered(0):
            success = True
            break
            
    if not success:
        # Fallback: try a smaller clique? 
        # Or just return the even numbers with a simple residue assignment that might work?
        # For even numbers, gcd >= 2. If we assign r_m = m % 2 == 0? No.
        # If we assign r_m = 1 for all, they intersect.
        # If we assign r_m = m, then r_i - r_j = m_i - m_j. 
        # gcd(m_i, m_j) divides m_i - m_j? Yes, always. 
        # So r_m = m always intersects? 
        # Wait. (m_i - m_j) % gcd(m_i, m_j) == 0. Yes. 
        # So we need r_m such that it is NOT congruent to others mod gcd.
        
        # If backtracking fails, it means no valid residue assignment exists for this clique?
        # That shouldn't happen for even numbers. 
        # Example: m1=2, m2=4. gcd=2. r1=0, r2=1. 1-0=1, 1%2!=0. OK.
        # m1=2, m2=6. gcd=2. r1=0, r2=1. OK.
        # m1=4, m2=6. gcd=2. r1=0, r2=1. OK.
        # It is likely always solvable for the clique of evens.
        
        # If we still fail, return a minimal valid set
        return [[1, 2]]
        
    result = []
    for m in selected_moduli:
        result.append([assignment[m], m])
        
    return result

# EVOLVE-BLOCK-END
