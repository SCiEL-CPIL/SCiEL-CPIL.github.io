import time
import math
from fractions import Fraction

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit

    # All candidate (m, r) pairs: m in [20, 200], r in [0, m)
    candidates = []
    for m in range(20, 201):
        for r in range(m):
            candidates.append((m, r))

    n = len(candidates)
    # Build conflict matrix: conflict[i][j] = True if (m_i, r_i) conflicts with (m_j, r_j)
    # Two candidates conflict if:
    #   1. Same modulus (can't pick two residues for same m)
    #   2. r_i % gcd(m_i, m_j) == r_j % gcd(m_i, m_j) (not disjoint)
    # Note: if gcd(m_i, m_j) == 1, then r_i % 1 = 0 and r_j % 1 = 0, so they conflict? 
    # Wait - the condition is r1 != r2 modulo gcd(m1, m2). If gcd=1, then r1 mod 1 = 0 and r2 mod 1 = 0,
    # so r1 % gcd == r2 % gcd, meaning they ARE compatible (disjoint). 
    # The conflict condition is: r_i % gcd(m_i, m_j) == r_j % gcd(m_i, m_j) means they are NOT disjoint (conflict).
    # If gcd = 1, r_i % 1 = 0 = r_j % 1, so they conflict? No wait...
    # Re-reading: "pairwise disjoint (r_1 != r_2 modulo gcd(m_1, m_2))"
    # So the constraint is r_1 != r_2 (mod gcd(m_1, m_2)).
    # If gcd = 1, then r_1 mod 1 = 0 and r_2 mod 1 = 0, so r_1 == r_2 mod 1, which violates the constraint.
    # Wait, that means gcd=1 pairs ALWAYS conflict? But the knowledge says "gcd=1 pairs were incorrectly rejected (they're always compatible since any residues are distinct mod 1)".
    # Let me re-read: "r_1 != r_2 modulo gcd(m_1, m_2)" - if gcd=1, everything is 0 mod 1, so r_1 mod 1 = r_2 mod 1 = 0, which means r_1 == r_2 mod 1, violating !=.
    # But the knowledge note says they're always compatible. Let me think again...
    # Actually "r_1 != r_2 modulo g" means the residue classes are disjoint as sets. Two residue classes r1 mod m1 and r2 mod m2 are disjoint iff r1 != r2 (mod gcd(m1,m2)).
    # If gcd(m1,m2)=1, then there's no solution to x ≡ r1 (mod m1) and x ≡ r2 (mod m2) being the same... wait no, CRT says there IS a unique solution mod m1*m2.
    # Hmm, let me reconsider. Two arithmetic progressions r1 + k*m1 and r2 + l*m2 are disjoint iff r1 != r2 (mod gcd(m1,m2)).
    # If gcd(m1,m2)=1, then r1 mod 1 = 0 and r2 mod 1 = 0, so r1 == r2 (mod 1), which means they ARE NOT disjoint? That can't be right.
    # Actually wait - if gcd(m1,m2)=1, the two residue classes intersect at exactly one point mod m1*m2. So they are NOT disjoint.
    # But the knowledge says "gcd=1 pairs were incorrectly rejected (they're always compatible)". Let me re-read the original constraint.
    # "pairwise disjoint (r_1 != r_2 modulo gcd(m_1, m_2))"
    # So the condition for being VALID (disjoint) is r_1 != r_2 (mod gcd(m_1, m_2)).
    # If gcd=1: r_1 mod 1 = 0, r_2 mod 1 = 0, so 0 != 0 is FALSE. So gcd=1 pairs are NOT disjoint.
    # But the measured knowledge says they improved by fixing "gcd=1 pairs were incorrectly rejected (they're always compatible since any residues are distinct mod 1)".
    # I think there might be confusion. Let me just follow the stated constraint literally:
    # Valid iff r1 % gcd(m1,m2) != r2 % gcd(m1,m2) for all pairs.
    # Conflict (invalid) iff r1 % gcd(m1,m2) == r2 % gcd(m1,m2).
    # For gcd=1: r1%1==0==r2%1, so conflict. So gcd=1 pairs conflict.
    # But the knowledge says the opposite... I'll trust the literal constraint from the task description.
    
    # Actually, re-reading more carefully: the seed program uses moduli 20,40,60,...,200 which all share gcd 20, and residues 0,1,...,9.
    # gcd(20,40)=20, 0%20=0, 1%20=1, 0!=1 so OK. gcd(40,60)=20, 1%20=1, 2%20=2, OK.
    # Let me just build the conflict graph properly.
    
    # For efficiency, precompute gcds
    # candidates[i] = (m_i, r_i)
    # conflict[i][j] if same m OR r_i % gcd(m_i, m_j) == r_j % gcd(m_i, m_j)
    
    # This is a large graph (n ~ sum of m for m in 20..200 = ~20000 nodes)
    # Too large for exact max closure. Let's use a different approach.
    
    # Greedy + local search approach:
    # 1. Sort candidates by 1/m (descending, i.e., m ascending) - prefer small moduli
    # 2. Greedily add compatible candidates
    # 3. Local search: try swapping or adding/removing
    
    # Actually, let's think about what's optimal. We want to maximize sum 1/m.
    # Smaller m gives larger 1/m, so we prefer small moduli. But we need pairwise disjointness.
    
    # Key insight: if we pick moduli that are all multiples of some base, we can control disjointness.
    # The seed picks m=20,40,...,200 (10 moduli) with residues 0-9, score = sum 1/(20j) for j=1..10 = (1/20)*H(10) ≈ 0.05*2.929 = 0.14645
    
    # Can we do better? What about using all 181 moduli from 20 to 200? No, most pairs will conflict.
    
    # Let me use a smarter approach:
    # 1. Try greedy with different orderings
    # 2. Local search with swaps
    
    # For a given set of moduli, we need to find compatible residues.
    # This is a constraint satisfaction / graph coloring problem on the conflict graph.
    
    # Let me try: start with a good set of moduli, then solve for residues.
    
    # Strategy: 
    # - Consider subsets of moduli from [20,200]
    # - For a given set S of moduli, find if there exist residues r_m for m in S such that all are pairwise disjoint
    # - Maximize sum 1/m over feasible S
    
    # This is still hard. Let me use a practical approach:
    # 1. Greedy construction with multiple random restarts
    # 2. Local search improvement
    
    import random
    
    def is_compatible(selected, m, r):
        """Check if (m,r) is compatible with all in selected (dict m->r)."""
        for m2, r2 in selected.items():
            if m2 == m:
                continue
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True
    
    def greedy_construct(order=None, seed=None):
        if seed is not None:
            random.seed(seed)
        selected = {}
        candidates = list(range(20, 201))
        if order == 'ascending':
            candidates.sort()
        elif order == 'descending':
            candidates.sort(reverse=True)
        elif order == 'random':
            random.shuffle(candidates)
        else:
            candidates.sort()  # ascending = prefer small m = large 1/m
        
        for m in candidates:
            if m in selected:
                continue
            # Try residues in random order or sequential
            residues = list(range(m))
            if order == 'random':
                random.shuffle(residues)
            else:
                # Try residue 0 first, then others
                pass
            for r in residues:
                if is_compatible(selected, m, r):
                    selected[m] = r
                    break
        return selected
    
    def score(selected):
        return sum(Fraction(1, m) for m in selected)
    
    def local_search(selected, max_iters=5000, deadline=None):
        """Improve via add, remove, and swap operations."""
        best = dict(selected)
        best_score = score(best)
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            improved = False
            # Try adding a new modulus
            current_moduli = set(best.keys())
            candidates_to_add = [m for m in range(20, 201) if m not in current_moduli]
            random.shuffle(candidates_to_add)
            
            for m in candidates_to_add[:20]:  # limit attempts per iteration
                found = False
                for r in range(m):
                    if is_compatible(best, m, r):
                        new_sel = dict(best)
                        new_sel[m] = r
                        new_score = score(new_sel)
                        if new_score > best_score:
                            best = new_sel
                            best_score = new_score
                            improved = True
                            found = True
                            break
                if found:
                    break
            
            # Try swapping: remove one, add another
            if not improved and len(best) > 1:
                items = list(best.items())
                random.shuffle(items)
                for m_remove, r_remove in items[:5]:
                    candidates_to_add = [m for m in range(20, 201) if m not in best]
                    random.shuffle(candidates_to_add)
                    for m_add in candidates_to_add[:15]:
                        new_sel = {m: r for m, r in best.items() if m != m_remove}
                        for r_add in range(m_add):
                            if is_compatible(new_sel, m_add, r_add):
                                new_sel[m_add] = r_add
                                new_score = score(new_sel)
                                if new_score > best_score:
                                    best = new_sel
                                    best_score = new_score
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                # Try re-optimizing residues for existing moduli
                # Pick a random modulus and try changing its residue
                if best:
                    m_test = random.choice(list(best.keys()))
                    r_old = best[m_test]
                    for r_new in range(20):  # try first 20 residues
                        if r_new == r_old:
                            continue
                        if is_compatible({m: r for m, r in best.items() if m != m_test}, m_test, r_new):
                            new_sel = dict(best)
                            new_sel[m_test] = r_new
                            new_score = score(new_sel)
                            if new_score > best_score:
                                best = new_sel
                                best_score = new_score
                                improved = True
                                break
                    # Also try larger residues
                    if not improved:
                        for r_new in range(m_test):
                            if r_new == r_old:
                                continue
                            if is_compatible({m: r for m, r in best.items() if m != m_test}, m_test, r_new):
                                new_sel = dict(best)
                                new_sel[m_test] = r_new
                                new_score = score(new_sel)
                                if new_score > best_score:
                                    best = new_sel
                                    best_score = new_score
                                    improved = True
                                    break
            
            if not improved:
                break
        
        return best
    
    # Run multiple strategies
    best_overall = {}
    best_overall_score = Fraction(0)
    
    # Strategy 1: Greedy ascending with different seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        sel = greedy_construct(order='ascending', seed=seed)
        s = score(sel)
        if s > best_overall_score:
            best_overall = sel
            best_overall_score = s
    
    # Strategy 2: Greedy with random order
    for seed in range(20):
        if time.time() > deadline:
            break
        sel = greedy_construct(order='random', seed=seed)
        s = score(sel)
        if s > best_overall_score:
            best_overall = sel
            best_overall_score = s
    
    # Strategy 3: Local search from best greedy
    if time.time() < deadline - 10:
        improved = local_search(best_overall, max_iters=3000, deadline=deadline - 5)
        s = score(improved)
        if s > best_overall_score:
            best_overall = improved
            best_overall_score = s
    
    # Strategy 4: Try specific structured constructions
    # Consecutive multiples
    for base in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
        if time.time() > deadline:
            break
        sel = {}
        for m in range(base, 201, base):
            # Find compatible residue
            for r in range(m):
                if is_compatible(sel, m, r):
                    sel[m] = r
                    break
        s = score(sel)
        if s > best_overall_score:
            best_overall = sel
            best_overall_score = s
    
    # Try coprime-friendly sets: moduli that are pairwise coprime or have small gcd
    # For now, also try the original seed construction as a baseline
    seed_sel = {20*j: j-1 for j in range(1, 11)}
    seed_score = score(seed_sel)
    if seed_score > best_overall_score:
        best_overall = seed_sel
        best_overall_score = seed_score
    
    # Final local search polish
    if time.time() < deadline - 3:
        final = local_search(best_overall, max_iters=2000, deadline=deadline - 1)
        s = score(final)
        if s > best_overall_score:
            best_overall = final
            best_overall_score = s
    
    return [[m, r] for m, r in sorted(best_overall.items())]
