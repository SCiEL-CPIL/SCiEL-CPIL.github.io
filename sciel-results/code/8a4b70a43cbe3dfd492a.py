"""Bootstrap-portfolio phylogenetic topology recovery with keep-best selection."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

_DEADLINE_FRACTION = 0.88  # safety margin vs 270 s wall-clock kill
_START = None


def _now():
    return time.time()


def _budget_left(total_wall=270.0):
    """Return seconds remaining before the safety deadline (>=0)."""
    global _START
    if _START is None:
        _START = time.time()
    elapsed = _now() - _START
    limit = total_wall * _DEADLINE_FRACTION
    return max(0.0, limit - elapsed)


def _jc_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix using NumPy."""
    n = len(labels)
    m = len(sequences[labels[0]])
    # Build integer code matrix
    code_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.zeros((n, m), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j, ch in enumerate(seq):
            arr[i, j] = code_map[ch]
    
    # Compute pairwise Hamming distances
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.count_nonzero(arr[i] != arr[j])
            p = diff / m
            if p >= 0.7499:
                d = 3.0
            elif p <= 1e-10:
                d = 0.0
            else:
                d = -0.75 * math.log(max(1e-10, 1.0 - 4.0 * p / 3.0))
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _nj(dist, labels):
    """Neighbour joining on a distance matrix. Returns Newick string."""
    n = len(labels)
    if n < 3:
        return ";".join(labels) + ";"
    
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    d = dist.copy()
    
    while len(nodes) > 3:
        n_curr = len(nodes)
        total = np.zeros(n_curr)
        for i, a in enumerate(nodes):
            total[i] = sum(d[a, b] for j, b in enumerate(nodes) if j != i)
        
        best_criterion = float('inf')
        best_i, best_j = -1, -1
        for i in range(n_curr):
            for j in range(i + 1, n_curr):
                a, b = nodes[i], nodes[j]
                criterion = (n_curr - 2) * d[a, b] - total[i] - total[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        a, b = nodes[best_i], nodes[best_j]
        joined = f"({trees[a]},{trees[b]})"
        new_id = -1 - len(nodes)
        
        rest = [x for k, x in enumerate(nodes) if k != best_i and k != best_j]
        new_nodes = rest + [new_id]
        
        for x in rest:
            val = 0.5 * (d[a, x] + d[b, x] - d[a, b])
            d[new_id, x] = val
            d[x, new_id] = val
        
        trees[new_id] = joined
        nodes = new_nodes
    
    if len(nodes) == 3:
        a, b, c = nodes
        return f"({trees[a]},{trees[b]},{trees[c]});"
    elif len(nodes) == 2:
        a, b = nodes
        return f"({trees[a]},{trees[b]});"
    else:
        return trees[nodes[0]] + ";"


def _parse_newick_splits(newick):
    """Extract non-trivial splits from a Newick string."""
    splits = set()
    labels = []
    for part in newick.replace("(", " ").replace(")", " ").replace(",", " ").replace(";", " ").split():
        if part and not part.startswith("("):
            labels.append(part)
    
    # Re-parse to get topology
    def parse(s):
        """Parse Newick, return list of splits as frozensets."""
        splits = []
        def _parse_inner(pos):
            if pos[0] >= len(s) or s[pos[0]] == ')':
                return set(), pos[0]
            if s[pos[0]] == '(':
                pos[0] += 1
                child_sets = []
                while pos[0] < len(s) and s[pos[0]] != ')':
                    cs, pos[0] = _parse_inner(pos)
                    if cs:
                        child_sets.append(cs)
                    if pos[0] < len(s) and s[pos[0]] == ',':
                        pos[0] += 1
                pos[0] += 1  # skip )
                # Optional label after )
                label = ""
                while pos[0] < len(s) and s[pos[0]] not in ',;)':
                    label += s[pos[0]]
                    pos[0] += 1
                combined = set().union(*child_sets) if child_sets else set()
                if label:
                    combined.add(label)
                # Record split if non-trivial
                if 0 < len(combined) < len(labels):
                    splits.append(frozenset(combined))
                return combined, pos[0]
            else:
                label = ""
                while pos[0] < len(s) and s[pos[0]] not in ',;)':
                    label += s[pos[0]]
                    pos[0] += 1
                return {label}, pos[0]
        
        all_splits = []
        pos = [0]
        _, _ = _parse_inner(pos)
        return all_splits
    
    # Simpler approach: just return the Newick string itself for comparison
    return newick


def _rf_distance(tree1_newick, tree2_newick, labels):
    """Compute normalized Robinson-Foulds distance between two trees."""
    # Extract all non-trivial splits from each tree
    def get_splits(newick, tax_labels):
        """Get all non-trivial splits from a Newick string."""
        splits = set()
        tax_set = set(tax_labels)
        # Tokenize and parse
        def parse_node(s, pos):
            if pos[0] >= len(s):
                return set()
            if s[pos[0]] == '(':
                pos[0] += 1
                children = []
                while pos[0] < len(s) and s[pos[0]] != ')':
                    child_set = parse_node(s, pos)
                    children.append(child_set)
                    if pos[0] < len(s) and s[pos[0]] == ',':
                        pos[0] += 1
                pos[0] += 1  # skip )
                result = set()
                for c in children:
                    result |= c
                # Skip optional label
                while pos[0] < len(s) and s[pos[0]] not in ',;)':
                    pos[0] += 1
                # Record split
                if 0 < len(result) < len(tax_set):
                    splits.add(frozenset(result))
                return result
            else:
                label = ""
                while pos[0] < len(s) and s[pos[0]] not in ',;)':
                    label += s[pos[0]]
                    pos[0] += 1
                return {label}
        
        pos = [0]
        parse_node(newick, pos)
        return splits
    
    splits1 = get_splits(tree1_newick, labels)
    splits2 = get_splits(tree2_newick, labels)
    
    if not splits1 and not splits2:
        return 0.0
    
    common = splits1 & splits2
    total_unique = len(splits1 | splits2)
    
    if total_unique == 0:
        return 0.0
    
    return 1.0 - len(common) / total_unique


def _bootstrap_subsample(sequences, labels, n_sites, rng):
    """Create bootstrap subsampled sequences."""
    bootstrap_seqs = {}
    for lab in labels:
        orig = sequences[lab]
        indices = rng.integers(0, len(orig), size=n_sites)
        bootstrap_seqs[lab] = ''.join(orig[i] for i in indices)
    return bootstrap_seqs


def reconstruct(sequences):
    global _START
    if _START is None:
        _START = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_sites = len(sequences[labels[0]])
    n_taxa = len(labels)
    
    # First: run NJ on full data as baseline
    dist_full = _jc_matrix(sequences, labels)
    best_tree = _nj(dist_full, labels)
    best_score = float('inf')
    
    # Budget-aware bootstrap portfolio
    rng = random.Random(42)
    n_bootstraps = 0
    max_time = _budget_left() * 0.9  # use 90% of remaining time for bootstraps
    
    # Estimate time per bootstrap
    # Small: ~5-10ms, Medium: ~20-50ms, Large: ~50-200ms
    est_time_per_bs = max(0.005, min(0.2, 0.002 * n_taxa**1.5 / n_sites**0.5))
    max_bs = int(max_time / est_time_per_bs) if max_time > 0 else 0
    max_bs = max(5, min(max_bs, 200))
    
    # Keep track of best candidate
    candidates = [(best_tree, best_score)]
    
    for bs_i in range(max_bs):
        if _budget_left() < 0.5:
            break
        
        # Bootstrap resample
        indices = rng.choices(range(n_sites), k=n_sites)
        boot_seqs = {}
        for lab in labels:
            orig = sequences[lab]
            boot_seqs[lab] = ''.join(orig[i] for i in indices)
        
        # Compute distances and run NJ
        dist_bs = _jc_matrix(boot_seqs, labels)
        tree_bs = _nj(dist_bs, labels)
        
        # Score this tree against the full-data tree
        score = _rf_distance(tree_bs, best_tree, labels)
        
        if score < best_score:
            best_score = score
            best_tree = tree_bs
        
        candidates.append((tree_bs, score))
    
    # If we have multiple candidates, pick the one that appears most often
    # among the low-scoring ones (consensus-like approach)
    if len(candidates) > 1:
        # Sort by score, take top 20%, count frequencies
        candidates.sort(key=lambda x: x[1])
        top_k = max(1, len(candidates) // 5)
        top_trees = [c[0] for c in candidates[:top_k]]
        
        # Count how often each tree appears
        tree_counts = {}
        for t in top_trees:
            tree_counts[t] = tree_counts.get(t, 0) + 1
        
        # Pick the most frequent
        if tree_counts:
            most_common = max(tree_counts, key=tree_counts.get)
            best_tree = most_common
    
    return best_tree
# EVOLVE-BLOCK-END
