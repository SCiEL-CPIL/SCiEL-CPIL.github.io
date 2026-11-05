import math

def construct():
    """
    Constructs a set of pairwise disjoint residue classes with distinct moduli in [20, 200]
    to maximize the sum of reciprocals.
    
    Strategy:
    Select a set of moduli that are pairwise coprime. This ensures that for any pair (m_i, m_j),
    gcd(m_i, m_j) = 1, which means the condition r_i != r_j (mod 1) is always satisfied
    (since there is only one residue class mod 1). Thus, we can set all residues r_i = 0.
    
    To maximize the sum of 1/m_i, we greedily select the smallest integers in [20, 200]
    that are coprime to all previously selected integers.
    """
    selected_moduli = []
    
    for m in range(20, 201):
        is_coprime_to_all = True
        for m_sel in selected_moduli:
            if math.gcd(m, m_sel) != 1:
                is_coprime_to_all = False
                break
        if is_coprime_to_all:
            selected_moduli.append(m)
            
    # Create the classes with residue 0
    classes = [[m, 0] for m in selected_moduli]
    
    return classes
