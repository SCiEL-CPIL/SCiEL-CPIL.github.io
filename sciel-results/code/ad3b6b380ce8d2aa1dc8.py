"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # All candidate moduli in [20, 200]
    all_moduli = list(range(20, 201))
    
    def is_compatible(m_new, r_new, selected):
        """Check if (m_new, r_new) is disjoint from all (m_old, r_old) in selected."""
        for m_old, r_old in selected:
            g = math.gcd(m_new, m_old)
            if r_new % g == r_old % g:
                return False
        return True
    
    def get_valid_residues(m_new, selected):
        """Get all valid residues for m_new given selected classes."""
        valid = []
        for r in range(m_new):
            if is_compatible(m_new, r, selected):
                valid.append(r)
        return valid
    
    def greedy_build(moduli_order, randomize_residues=False):
        """
        Greedily build a set of disjoint residue classes.
        Process moduli in the given order; for each, pick a valid residue.
        If randomize_residues, pick randomly among valid residues; else pick smallest.
        Returns (classes, score) where score is sum of 1/m.
        """
        selected = []
        score = Fraction(0, 1)
        for m in moduli_order:
            valid = get_valid_residues(m, selected)
            if not valid:
                continue
            if randomize_residues:
                r = random.choice(valid)
            else:
                r = valid[0]
            selected.append((m, r))
            score += Fraction(1, m)
        return selected, score
    
    def run_greedy_randomized(num_restarts):
        """
        Run multiple randomized greedy attempts.
        Each attempt: shuffle moduli, then greedily add with random residue choice.
        Track best.
        """
        best_classes = []
        best_score = Fraction(0, 1)
        
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
            moduli_order = all_moduli[:]
            random.shuffle(moduli_order)
            classes, score = greedy_build(moduli_order, randomize_residues=True)
            if score > best_score:
                best_score = score
                best_classes = classes[:]
        
        return best_classes, best_score
    
    def run_deterministic_greedy():
        """
        Deterministic greedy: process moduli in increasing order, pick smallest valid residue.
        """
        classes, score = greedy_build(all_moduli, randomize_residues=False)
        return classes, score
    
    # First, run deterministic greedy to get a baseline
    det_classes, det_score = run_deterministic_greedy()
    best_classes = det_classes[:]
    best_score = det_score
    
    # Now run randomized greedy with many restarts
    # Estimate: each greedy run processes up to 181 moduli, each checking against selected.
    # This should be fast, maybe 0.001-0.01s per run. Let's do as many as we can.
    
    num_restarts = 5000
    rand_classes, rand_score = run_greedy_randomized(num_restarts)
    if rand_score > best_score:
        best_score = rand_score
        best_classes = rand_classes[:]
    
    # Convert to required format: [[m, r], ...]
    result = [[m, r] for m, r in best_classes]
    return result
# EVOLVE-BLOCK-END
