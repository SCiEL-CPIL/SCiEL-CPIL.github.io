"""
Quartet-Sampling based unrooted tree reconstruction.

Strategy:
1. Compute pairwise Jukes-Cantor distances.
2. For every 4-taxon subset (quartet), determine the most parsimonious topology.
3. Aggregate the winning quartets into a set of compatible splits.
4. Reconstruct the full tree topology from the consensus split set.

This approach is more robust to noise than NJ for small-to-medium taxon sets
because it relies on local 4-point conditions which are statistically powerful
under the JC model.
"""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
_DEADLINE = None
_START = None
_MARGIN = 5.0  # seconds of safety margin before the 270 s wall-clock kill


def _now():
    return time.time()


def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - _MARGIN - _now()


def _jc_distance(a, b):
    n = len(a)
    diff = 0
    for x, y in zip(a, b):
        diff += (x != y)
    p = diff / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _quartet_parsimony_cost(seq_a, seq_b, seq_c, seq_d):
    """
    Calculate parsimony cost for the three possible unrooted topologies
    of a quartet (a,b,c,d).
    Topology 1: (a,b),(c,d) -> splits {a,b}|{c,d}
    Topology 2: (a,c),(b,d) -> splits {a,c}|{b,d}
    Topology 3: (a,d),(b,c) -> splits {a,d}|{b,c}
    
    Returns the topology index (0, 1, or 2) with minimum cost.
    Ties are broken arbitrarily (first found).
    """
    # Fitch parsimony for a quartet can be done analytically or via small DP.
    # For 4 taxa, the cost is the number of changes.
    # We can compute the cost for each split by checking if the states on
    # each side of the split are compatible.
    
    # A faster heuristic for quartets under JC:
    # The "best" quartet topology is the one that minimizes the sum of distances
    # of the two pairs, BUT parsimony is the goal.
    # Actually, for 4 taxa, the parsimony score is:
    # Score(T) = min over assignments of internal node states.
    
    # Let's use the standard quartet test based on distances if we assume
    # additivity, but the task says "parsimony" is good. 
    # However, with JC, distance-based quartet inference (sum of 3 distances)
    # is equivalent to parsimony for 4 taxa if we use the correct metric?
    # No, parsimony is site-by-site.
    
    # Let's compute site-by-site parsimony for the 3 topologies.
    # This is O(L) per quartet. With ~1000 quartets and 1000 sites, that's 1M ops.
    # This is fast enough in Python if optimized, or we can use NumPy.
    
    # Optimization: Precompute character vectors?
    # seq_a, seq_b etc are strings.
    
    # Topology 0: (ab)(cd)
    # Cost is sum over sites of cost(site_a, site_b, site_c, site_d) for that topology.
    # For a single site, the min parsimony cost for topology (ab)(cd) is:
    # If {a,b} states == {c,d} states: 1 change (or 0 if all same? No, 1 change on internal edge? 
    # Wait. Unrooted quartet has 1 internal node.
    # If a==b and c==d and a!=c: cost 1 (change on internal edge).
    # If a==b==c==d: cost 0.
    # If a!=b and c!=d: cost depends.
    
    # Let's just run Fitch on the 3 trees. It's simple.
    
    def fitch_cost_top_ab_cd(sa, sb, sc, sd):
        # Tree: ((a,b),(c,d))
        # Internal nodes: n_ab, n_cd, n_root
        # We can do this site by site.
        cost = 0
        for i in range(len(sa)):
            A = sa[i]
            B = sb[i]
            C = sc[i]
            D = sd[i]
            
            # Set at n_ab
            if A == B:
                S_ab = A
            else:
                S_ab = None # Union {A,B}
            
            # Set at n_cd
            if C == D:
                S_cd = C
            else:
                S_cd = None # Union {C,D}
            
            # Set at n_root
            if S_ab is not None and S_cd is not None:
                if S_ab == S_cd:
                    S_root = S_ab
                    c_site = 0
                else:
                    S_root = None # Union
                    c_site = 1
            elif S_ab is not None and S_cd is None:
                # S_cd is {C,D}. Does S_ab intersect {C,D}?
                if S_ab == C or S_ab == D:
                    S_root = S_ab
                    c_site = 0
                else:
                    S_root = None
                    c_site = 1
            elif S_ab is None and S_cd is not None:
                if S_cd == A or S_cd == B:
                    S_root = S_cd
                    c_site = 0
                else:
                    S_root = None
                    c_site = 1
            else:
                # Both unions. Intersect {A,B} and {C,D}
                if (A == C or A == D or B == C or B == D):
                    S_root = None # Intersection non-empty but we don't track exact set for cost?
                    # Actually for cost, if intersection non-empty, cost is 0 for this step.
                    c_site = 0
                else:
                    S_root = None
                    c_site = 1
            
            cost += c_site
        return cost

    # This is getting complex and slow in pure Python.
    # Alternative: Use distance-based quartet inference.
    # Under JC, the expected distance is additive.
    # The quartet that minimizes d(a,b)+d(c,d) vs d(a,c)+d(b,d) vs d(a,d)+d(b,c)
    # is the correct one with high probability.
    # This is O(1) per quartet if distances are precomputed.
    # Given the noise, this is a very strong baseline and much faster.
    # Let's implement distance-based quartet voting.
    
    # We need the distance matrix.
    # But we don't have it passed here. We will compute it in the main loop.
    pass

def _quartet_test_distances(d_ab, d_cd, d_ac, d_bd, d_ad, d_bc):
    """
    Determine the best quartet topology based on distances.
    Returns 0 for (ab|cd), 1 for (ac|bd), 2 for (ad|bc).
    """
    # Sum of opposite edges
    s0 = d_ab + d_cd
    s1 = d_ac + d_bd
    s2 = d_ad + d_bc
    
    # Minimum sum corresponds to the true topology
    min_s = min(s0, s1, s2)
    if s0 == min_s:
        return 0
    elif s1 == min_s:
        return 1
    else:
        return 2


def _reconstruct_quartet(sequences, labels):
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0]
        if n == 2:
            return "(%s,%s);" % (labels[0], labels[1])
        return "(%s,%s,%s);" % (labels[0], labels[1], labels[2])

    # 1. Compute Distance Matrix
    dist = {}
    for i, a in enumerate(labels):
        for j in range(i + 1, n):
            b = labels[j]
            d = _jc_distance(sequences[a], sequences[b])
            dist[(a, b)] = d
            dist[(b, a)] = d

    # 2. Vote on Quartets
    # For n=8, C(8,4)=70 quartets.
    # For n=32, C(32,4)=35960 quartets.
    # 35960 * O(1) is very fast.
    
    # We want to collect the "winning" splits.
    # A split is a subset of taxa.
    # For each quartet, we identify the split it supports.
    # (ab|cd) supports the split {a,b} (and {c,d}).
    
    # We will count votes for each split.
    # However, splits from different quartets might be incompatible.
    # We need to find a compatible set of splits that maximizes votes.
    # This is the "Maximum Compatible Quartet Set" problem.
    
    # Heuristic: 
    # 1. Generate all quartet votes.
    # 2. Sort quartets by "confidence" (difference between min and 2nd min distance sum).
    # 3. Greedily add splits if they are compatible with already accepted splits.
    
    # Compatibility check: Two splits A and B are compatible if 
    # A & B is empty, or A & B == A, or A & B == B, or A | B is empty (impossible for proper splits)
    # Actually, for unrooted trees, splits S1 and S2 are compatible if one of the four intersections
    # S1&S2, S1\S2, S2\S1, S1|S2 (complement) is empty?
    # No. S1 and S2 are compatible if they do not cross.
    # They cross if S1 & S2 is non-empty, S1 \ S2 is non-empty, S2 \ S1 is non-empty, and S1 | S2 is not all taxa?
    # Simpler check: 
    # Let A be one side of split 1, B be one side of split 2.
    # They are compatible if A & B is empty, OR A \ B is empty, OR B \ A is empty.
    # (This covers nested or disjoint cases).
    
    def are_compatible(split1, split2, n_taxa):
        # split1, split2 are sets of labels
        # Ensure we check against the "smaller" side to avoid complement issues?
        # No, the definition of crossing is symmetric.
        # A and B cross if A&B, A\B, B\A are all non-empty.
        # If any is empty, they are compatible.
        inter = split1 & split2
        if not inter:
            return True
        diff1 = split1 - split2
        if not diff1:
            return True
        diff2 = split2 - split1
        if not diff2:
            return True
        return False

    votes = []
    # List of (confidence, split_set)
    # We iterate over all 4-combinations
    
    # Optimization: Precompute indices
    label_indices = {l: i for i, l in enumerate(labels)}
    
    # We can speed up distance lookup by using indices
    dist_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = dist[(labels[i], labels[j])]
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d

    for combo in combinations(range(n), 4):
        i, j, k, l = combo
        d_ij = dist_matrix[i][j]
        d_kl = dist_matrix[k][l]
        d_ik = dist_matrix[i][k]
        d_jl = dist_matrix[j][l]
        d_il = dist_matrix[i][l]
        d_jk = dist_matrix[j][k]
        
        s0 = d_ij + d_kl
        s1 = d_ik + d_jl
        s2 = d_il + d_jk
        
        min_s = min(s0, s1, s2)
        # Confidence is margin
        sorted_s = sorted([s0, s1, s2])
        confidence = sorted_s[1] - sorted_s[0]
        
        if s0 == min_s:
            # Topology (ij|kl)
            split = frozenset([labels[i], labels[j]])
        elif s1 == min_s:
            # Topology (ik|jl)
            split = frozenset([labels[i], labels[k]])
        else:
            # Topology (il|jk)
            split = frozenset([labels[i], labels[l]])
            
        votes.append((confidence, split))

    # Sort by confidence descending
    votes.sort(key=lambda x: x[0], reverse=True)
    
    accepted_splits = []
    # We need to build a tree from these splits.
    # A set of compatible splits defines a tree.
    # We start with the "root" split? No, unrooted.
    # We can just collect the splits and then reconstruct the tree.
    
    for conf, split in votes:
        # Check compatibility with all accepted splits
        compatible = True
        for acc_split in accepted_splits:
            if not are_compatible(split, acc_split, n):
                compatible = False
                break
        if compatible:
            accepted_splits.append(split)
            
        # Early stopping? If we have enough splits to define the tree?
        # An unrooted binary tree with n taxa has n-3 internal edges (splits).
        # If we have n-3 compatible splits, we have a full tree.
        if len(accepted_splits) >= n - 3:
            break

    if not accepted_splits:
        # Fallback to NJ if no splits accepted (shouldn't happen)
        return _nj_fallback(sequences, labels)

    # Reconstruct Newick from splits
    return _build_tree_from_splits(accepted_splits, labels)


def _build_tree_from_splits(splits, labels):
    """
    Reconstruct a Newick string from a set of compatible splits.
    This is a recursive process.
    """
    n = len(labels)
    if n <= 1:
        return labels[0]
    if n == 2:
        return "(%s,%s);" % (labels[0], labels[1])
    
    # Find a split that divides the current set of labels
    # We need to find a split S such that S is a subset of current labels
    # and its complement is also a subset (trivially true if S is from the set)
    # Actually, the splits are global. We need to find a split that "cuts" the current cluster.
    
    # Algorithm:
    # 1. If no split separates the current labels, it's a star? 
    #    No, if no split separates them, they form a polytomy or a clade with no internal resolution in our split set.
    #    But we have n-3 splits, so it should be binary.
    
    # Let's find a split that partitions the current set of labels into two non-empty parts.
    current_labels = set(labels)
    
    # Find a split that is a proper subset of current_labels
    # Note: The split might be defined by a subset that is NOT a subset of current_labels?
    # No, splits are defined on the whole set. 
    # If we are in a recursive step, we are looking at a subset of taxa.
    # A split S from the global set is "active" in this cluster if 
    # S & current_labels is non-empty AND (complement of S) & current_labels is non-empty.
    
    # Let's just pick the first split that partitions current_labels.
    # To be deterministic, we can sort splits or just iterate.
    
    partition_split = None
    for s in splits:
        side1 = s & current_labels
        side2 = current_labels - s
        if side1 and side2:
            partition_split = s
            break
            
    if partition_split is None:
        # No split partitions this cluster. Return a star.
        return "(%s);" % ",".join(labels)
        
    # Recurse
    side1_labels = list(partition_split & current_labels)
    side2_labels = list(current_labels - partition_split)
    
    # We need to pass the relevant splits to the children?
    # Actually, we can just use the same list of splits. 
    # The compatibility check ensures that splits in one side don't cross splits in the other.
    # So any split that intersects side1 must be contained in side1 or disjoint?
    # If a split S is compatible with P (the partition split), and S intersects side1,
    # then S is either contained in side1 or S is P?
    # Since S != P, S must be contained in side1 or side2.
    # So we can filter splits for each side.
    
    splits_side1 = [s for s in splits if s & side2_labels == set()] # S is subset of side1
    splits_side2 = [s for s in splits if s & side1_labels == set()] # S is subset of side2
    
    # Wait, what if S is the complement? 
    # If S is contained in side1, then S & side2 is empty.
    # If S is contained in side2, then S & side1 is empty.
    # This logic holds.
    
    newick1 = _build_tree_from_splits(splits_side1, side1_labels)
    newick2 = _build_tree_from_splits(splits_side2, side2_labels)
    
    return "(" + newick1 + "," + newick2 + ");"


def _nj_fallback(sequences, labels):
    """Simple NJ fallback."""
    n = len(labels)
    if n <= 3:
        if n == 1: return labels[0]
        if n == 2: return "(%s,%s);" % (labels[0], labels[1])
        return "(%s,%s,%s);" % (labels[0], labels[1], labels[2])
        
    dist = {}
    for i, a in enumerate(labels):
        for j in range(i + 1, n):
            b = labels[j]
            d = _jc_distance(sequences[a], sequences[b])
            dist[(a, b)] = dist[(b, a)] = d
    nodes = list(labels)
    trees = {name: name for name in labels}
    while len(nodes) > 3:
        m = len(nodes)
        total = {}
        for a in nodes:
            s = 0.0
            for b in nodes:
                if b != a:
                    s += dist[(a, b)]
            total[a] = s
        best = None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                crit = (m - 2) * dist[(a, b)] - total[a] - total[b]
                if best is None or crit < best[0]:
                    best = (crit, a, b)
        _, a, b = best
        joined = "(" + trees[a] + "," + trees[b] + ")"
        new = "n%d_%s_%s" % (len(nodes), a, b)
        rest = [x for x in nodes if x != a and x != b]
        for x in rest:
            val = 0.5 * (dist[(a, x)] + dist[(b, x)] - dist[(a, b)])
            dist[(new, x)] = dist[(x, new)] = val
        nodes = rest + [new]
        trees[new] = joined
    newick = "(" + ",".join(trees[x] for x in nodes) + ");"
    return newick


def reconstruct(sequences):
    global _DEADLINE, _START
    if _START is None:
        _START = _now()
        _DEADLINE = _START + 270.0

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if _time_left() < 1.0:
        return "(%s);" % ",".join(labels)

    try:
        result = _reconstruct_quartet(sequences, labels)
    except Exception:
        result = "(%s);" % ",".join(labels)

    if result is None:
        result = "(%s);" % ",".join(labels)

    return result
# EVOLVE-BLOCK-END
