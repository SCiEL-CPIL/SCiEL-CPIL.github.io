"""Phylogenetic topology recovery: NJ with quartet refinement for small n."""
import math
import time
from itertools import combinations

_DEADLINE = 250.0  # safety margin before 270s kill
_START_TIME = None


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _jc_distance(a, b):
    """Jukes-Cantor corrected distance."""
    n = len(a)
    if n == 0:
        return 0.0
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _nj_topology(dist):
    """Standard Neighbor-Joining. dist is a dict {(i,j): d} for indices."""
    n = len(dist)
    if n <= 2:
        return dist
    # dist is indexed by 0..n-1
    nodes = list(range(n))
    trees = {i: f"t{i}" for i in range(n)}
    # Build full distance matrix
    d = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d[(i, j)] = dist[(min(i, j), max(i, j))]
    
    while len(nodes) > 3:
        m = len(nodes)
        # Compute row sums
        r = {}
        for a in nodes:
            r[a] = sum(d[(a, b)] for b in nodes if b != a)
        
        # Find best pair
        best = None
        for i_idx, a in enumerate(nodes):
            for b in nodes[i_idx + 1:]:
                criterion = (m - 2) * d[(a, b)] - r[a] - r[b]
                if best is None or criterion < best[0]:
                    best = (criterion, a, b)
        
        _, a, b = best
        # Create new node
        new_id = 10000 + len(nodes)  # unique id for new node
        joined = f"({trees[a]},{trees[b]})"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            # Barycenter formula for new distance
            dab = d[(min(a, b), max(a, b))]
            dax = d[(min(a, x), max(a, x))]
            dbx = d[(min(b, x), max(b, x))]
            new_d = 0.5 * (dax + dbx - dab)
            d[(min(new_id, x), max(new_id, x))] = new_d
        
        nodes = rest + [new_id]
        trees[new_id] = joined
    
    # Final 3 nodes form a star
    return "(" + ",".join(trees[x] for x in nodes) + ")"


def _splits_from_newick(newick, n_taxa):
    """Extract all non-trivial splits from a Newick string.
    Returns a set of frozensets representing splits."""
    # Parse Newick to get all internal nodes and their leaf sets
    # We need to handle the unrooted tree properly
    
    def parse(newick_str):
        """Parse Newick, return tree structure as nested lists."""
        # Remove trailing semicolon
        s = newick_str.strip().rstrip(';')
        # Find the top-level structure
        # For unrooted: ((a,b),(c,d)) means the tree has a central node with 3 branches
        # Actually in Newick, ((a,b),(c,d)) is a rooted tree with 2 children at root
        # For unrooted with n taxa, we have n-3 internal nodes
        
        # Let's use a stack-based parser
        # Return a tree where leaves are 't0'..'t(n-1)' and internal nodes are tuples
        
        tokens = []
        i = 0
        while i < len(s):
            c = s[i]
            if c == '(':
                tokens.append('(')
                i += 1
            elif c == ')':
                tokens.append(')')
                i += 1
            elif c == ',':
                tokens.append(',')
                i += 1
            elif c == ';':
                break
            else:
                j = i
                while j < len(s) and s[j] not in '(),;':
                    j += 1
                tokens.append(s[i:j])
                i = j
        
        # Parse tokens into tree
        def build(pos):
            if pos[0] >= len(tokens):
                return None
            if tokens[pos[0]] == '(':
                pos[0] += 1
                children = []
                while pos[0] < len(tokens) and tokens[pos[0]] != ')':
                    child = build(pos)
                    if child is not None:
                        children.append(child)
                    if pos[0] < len(tokens) and tokens[pos[0]] == ',':
                        pos[0] += 1
                    elif pos[0] < len(tokens) and tokens[pos[0]] == ')':
                        break
                if pos[0] < len(tokens) and tokens[pos[0]] == ')':
                    pos[0] += 1
                # Check for label after )
                if pos[0] < len(tokens) and tokens[pos[0]] not in '(),;':
                    label = tokens[pos[0]]
                    pos[0] += 1
                    return ('internal', children, label)
                return ('internal', children, None)
            else:
                # Leaf
                label = tokens[pos[0]]
                pos[0] += 1
                return ('leaf', label)
        
        pos = [0]
        tree = build(pos)
        return tree
    
    def get_leaves(node):
        """Get set of leaf labels under a node."""
        if node is None:
            return set()
        if node[0] == 'leaf':
            return {node[1]}
        else:
            leaves = set()
            for child in node[1]:
                leaves |= get_leaves(child)
            return leaves
    
    def collect_splits(node, parent_split=None):
        """Collect all non-trivial splits from the tree."""
        splits = set()
        if node is None:
            return splits
        if node[0] == 'leaf':
            return splits
        
        leaves = get_leaves(node)
        n_total = len(leaves) if parent_split is None else len(parent_split[0]) + len(parent_split[1])
        
        if node[0] == 'internal':
            # This node splits its leaves into two groups
            if len(node[1]) == 2:
                l1 = get_leaves(node[1][0])
                l2 = get_leaves(node[1][1])
                # The split is (l1, l2) if both are non-empty and not the whole set
                if len(l1) > 1 and len(l2) > 1:
                    # Normalize: sort the two parts
                    part1 = frozenset(l1)
                    part2 = frozenset(l2)
                    # Represent split as the smaller part (or first alphabetically)
                    if len(part1) < len(part2) or (len(part1) == len(part2) and sorted(part1) < sorted(part2)):
                        splits.add(part1)
                    else:
                        splits.add(part2)
            
            for child in node[1]:
                splits |= collect_splits(child)
        
        return splits
    
    tree = parse(newick)
    if tree is None:
        return set()
    
    # Get all leaves to know total
    all_leaves = get_leaves(tree)
    n_total = len(all_leaves)
    
    splits = set()
    
    def collect_splits_v2(node):
        """Collect splits more carefully."""
        if node is None or node[0] == 'leaf':
            return
        if node[0] == 'internal' and len(node[1]) == 2:
            l1 = get_leaves(node[1][0])
            l2 = get_leaves(node[1][1])
            if len(l1) > 1 and len(l2) > 1:
                # This is a non-trivial split
                # Represent as frozenset of the smaller side
                if len(l1) <= len(l2):
                    splits.add(frozenset(l1))
                else:
                    splits.add(frozenset(l2))
        
        for child in node[1]:
            collect_splits_v2(child)
    
    collect_splits_v2(tree)
    return splits


def _robinson_foulds(splits1, splits2):
    """Compute normalized RF distance between two sets of splits."""
    # Each split is represented as a frozenset (the smaller side)
    # Two splits match if one equals the other or one is the complement of the other
    # But since we always store the smaller side, we need to be careful
    
    # Actually, let's use a different representation:
    # A split is an unordered pair {A, B} where A ∪ B = all taxa
    # We can represent it as a frozenset of frozensets, but that's complex
    # Simpler: represent as frozenset(A) where A is the side with <= n/2 elements
    # Then two splits are the same if their stored frozensets are equal
    
    s1 = set(splits1)
    s2 = set(splits2)
    
    # Count matching splits
    # A split in s1 matches a split in s2 if they are the same frozenset
    # (since both are stored as the smaller side)
    matching = len(s1 & s2)
    
    total = len(s1) + len(s2) - matching
    if total == 0:
        return 0.0
    return (len(s1) + len(s2) - 2 * matching) / total


def _quartet_score(newick, dist):
    """Score a topology by summing quartet distances.
    For each quartet, check if the topology agrees with the distance matrix.
    Lower is better."""
    # Extract all quartets from the topology
    # This is expensive, so we'll use a simpler heuristic:
    # Sum of branch lengths implied by the topology vs distances
    
    # Actually, let's use a simpler approach:
    # For NJ, the topology is already determined. For refinement,
    # we'll try local rearrangements (NNI) and check if they improve
    # a distance-based objective.
    
    # For now, return 0 as placeholder
    return 0.0


def _nni_refine(newick, dist, n_taxa, max_iterations=50):
    """Try NNI moves to improve the topology.
    dist is a dict {(i,j): d} for indices 0..n-1."""
    # This is complex to implement correctly with Newick strings.
    # For small n, we can do exhaustive search instead.
    # For now, skip NNI and just return the input.
    return newick


def _exhaustive_best_topology(dist, n_taxa, deadline):
    """For small n (<=7), exhaustively search all unrooted topologies.
    Score each by the sum of squared differences between implied and observed distances.
    Returns the best Newick string."""
    if n_taxa <= 2:
        if n_taxa == 1:
            return "t0"
        return "(t0,t1)"
    
    if _elapsed() > deadline:
        return None
    
    # Generate all unrooted tree topologies
    # For n taxa, there are (2n-5)!! unrooted topologies
    # For n=5: 15, n=6: 105, n=7: 945
    
    # We'll represent a tree as a set of splits
    # Generate all valid sets of n-3 non-trivial splits
    
    # Simpler: enumerate all binary tree shapes
    # For n=5: 3 possible shapes (but with labeled leaves, 15 total)
    # For n=6: 105
    # For n=7: 945
    
    # Let's use a recursive generation of all unrooted binary trees
    # A tree can be represented by its splits
    
    # Actually, for n<=7, let's just enumerate all possible Newick strings
    # by considering all ways to combine pairs
    
    # Better approach: use the fact that an unrooted tree on n leaves
    # can be built by successively joining pairs
    # But that's what NJ does. For exhaustive, we need all topologies.
    
    # Let's generate all sets of n-3 splits that form a valid tree
    # A set of splits is valid if:
    # 1. Each split is non-trivial (both sides have >= 2 leaves)
    # 2. Any two splits are compatible (their four intersections are not all non-empty)
    # 3. There are exactly n-3 splits
    
    # For n=5, n-3=2 splits. Two splits on 5 leaves.
    # This is getting complex. Let's use a different approach.
    
    # For very small n (<=6), we can enumerate all possible Newick strings
    # by trying all combinations of which leaves are joined where.
    
    # Actually, the cleanest approach for n<=7 is to enumerate all
    # unrooted binary tree topologies using the standard enumeration.
    
    # For n=5: 3 topological types, each with 10 labelings = 15 total
    # For n=6: 6 topological types, each with 35 labelings = 105 total  
    # Wait, that's not right either.
    
    # Let me use a simpler method: for n<=6, try all possible
    # "quartet" combinations and build trees from them.
    
    # Actually, the simplest correct approach:
    # An unrooted tree on n leaves has n-3 internal edges.
    # We can enumerate all possible sets of n-3 compatible non-trivial splits.
    
    # For n=5: 2 splits needed. Total non-trivial splits on 5 leaves: C(5,2)*C(3,1)/2... 
    # This is getting complicated. Let me just use a practical approach.
    
    # For n <= 6, do a limited search:
    # 1. Start with NJ topology
    # 2. Try all possible quartet rearrangements
    # 3. Keep the best
    
    # For n=5, there are only 15 possible topologies, so we can enumerate all.
    # For n=6, 105 topologies, still feasible.
    # For n=7, 945 topologies, feasible within time budget.
    
    # Let's implement a proper enumeration.
    
    # A unrooted binary tree can be represented as a set of splits.
    # We'll generate all valid sets of n-3 splits.
    
    # First, generate all non-trivial splits
    all_splits = []
    for size in range(2, n_taxa // 2 + 1):
        for subset in combinations(range(n_taxa), size):
            s = frozenset(subset)
            all_splits.append(s)
    
    # Check compatibility of two splits
    def compatible(s1, s2):
        """Two splits are compatible if at least one of the four
        intersections is empty."""
        a = s1 & s2
        b = s1 - s2
        c = s2 - s1
        d = frozenset(range(n_taxa)) - s1 - s2
        # Actually, d = (all - s1) - s2 = all - s1 - s2
        return len(a) == 0 or len(b) == 0 or len(c) == 0 or len(d) == 0
    
    # Find all valid sets of n-3 compatible splits
    # This is exponential, but for n<=7 it's manageable
    
    target_splits = n_taxa - 3
    best_newick = None
    best_score = float('inf')
    
    # Use backtracking to find all valid split sets
    def backtrack(start_idx, current_splits, depth):
        nonlocal best_newick, best_score
        if _elapsed() > deadline:
            return
        if depth == target_splits:
            # Evaluate this topology
            score = _score_splits(current_splits, dist, n_taxa)
            if score < best_score:
                best_score = score
                best_newick = _splits_to_newick(current_splits, n_taxa)
            return
        
        for i in range(start_idx, len(all_splits)):
            s = all_splits[i]
            # Check compatibility with all current splits
            ok = True
            for cs in current_splits:
                if not compatible(s, cs):
                    ok = False
                    break
            if ok:
                current_splits.append(s)
                backtrack(i + 1, current_splits, depth + 1)
                current_splits.pop()
    
    backtrack(0, [], 0)
    
    if best_newick is not None:
        return best_newick
    return None


def _score_splits(splits, dist, n_taxa):
    """Score a set of splits by comparing implied distances to observed.
    Use the sum of squared errors as a proxy."""
    # For each pair of taxa, the implied distance is the sum of branch lengths
    # along the path. We need to estimate branch lengths from the splits.
    # This is complex. Instead, use a simpler heuristic:
    # For each quartet defined by the splits, check if it's consistent
    # with the distance matrix.
    
    # Simpler: count how many quartets are "correct"
    # A quartet (a,b,c,d) is correct if the topology groups them
    # in the way that minimizes total distance.
    
    total = 0.0
    for a, b, c, d in combinations(range(n_taxa), 4):
        # Three possible quartet topologies:
        # (a,b)|(c,d): d(a,b)+d(c,d) vs d(a,c)+d(b,d) vs d(a,d)+d(b,c)
        dab = dist[(min(a,b), max(a,b))]
        dac = dist[(min(a,c), max(a,c))]
        dad = dist[(min(a,d), max(a,d))]
        dbc = dist[(min(b,c), max(b,c))]
        dbd = dist[(min(b,d), max(b,d))]
        dcd = dist[(min(c,d), max(c,d))]
        
        # The three sums:
        s1 = dab + dcd  # (a,b)|(c,d)
        s2 = dac + dbd  # (a,c)|(b,d)
        s3 = dad + dbc  # (a,d)|(b,c)
        
        # The correct topology minimizes this sum
        min_sum = min(s1, s2, s3)
        
        # Check which topology our splits imply
        # For quartet a,b,c,d, look at the splits to determine grouping
        # A split {a,b} means a and b are on the same side
        # If there's a split separating {a,b} from {c,d}, then topology is (a,b)|(c,d)
        
        # Determine the quartet topology from splits
        # For each pair, check if they're separated by any split
        def separated(x, y):
            for s in splits:
                if x in s and y not in s:
                    return True
                if x not in s and y in s:
                    return True
            return False
        
        # The quartet topology is determined by which pairs are separated
        # In a correct quartet, exactly 2 pairs are separated (the "long" pairs)
        # and 2 are not (the "short" pairs within each group)
        
        ab_sep = separated(a, b)
        ac_sep = separated(a, c)
        ad_sep = separated(a, d)
        bc_sep = separated(b, c)
        bd_sep = separated(b, d)
        cd_sep = separated(c, d)
        
        # Count separations
        # If topology is (a,b)|(c,d): ab not sep, cd not sep, others sep
        # If topology is (a,c)|(b,d): ac not sep, bd not sep, others sep
        # If topology is (a,d)|(b,c): ad not sep, bc not sep, others sep
        
        # Score: 0 if matches the min-sum topology, 1 otherwise
        if s1 == min_sum:
            # Expected: ab not sep, cd not sep
            if not ab_sep and not cd_sep:
                total += 0.0
            else:
                total += 1.0
        elif s2 == min_sum:
            if not ac_sep and not bd_sep:
                total += 0.0
            else:
                total += 1.0
        else:
            if not ad_sep and not bc_sep:
                total += 0.0
            else:
                total += 1.0
    
    return total


def _splits_to_newick(splits, n_taxa):
    """Convert a set of splits to a Newick string."""
    # Build the tree from splits
    # Start with all leaves as separate components
    # Each split merges two components
    
    # Represent the tree as a graph
    # Nodes: leaves 0..n-1 and internal nodes
    # Edges connect nodes that are in the same part of every split
    
    # Simpler: build the tree recursively
    # The "root" of the unrooted tree is arbitrary
    # Pick a split and use it as the top level
    
    if not splits:
        if n_taxa == 1:
            return "t0"
        elif n_taxa == 2:
            return "(t0,t1)"
        else:
            # Star tree
            return "(" + ",".join(f"t{i}" for i in range(n_taxa)) + ")"
    
    # Build adjacency: two taxa are adjacent in the "split graph" if
    # no split separates them. But that's not quite right for the tree.
    
    # Better: the tree can be reconstructed from splits.
    # Two internal edges correspond to two splits.
    # The tree has n leaves and n-3 internal nodes.
    
    # Let's build the tree by finding the hierarchical structure.
    # Sort splits by size (smaller first)
    sorted_splits = sorted(splits, key=len)
    
    # Build a hierarchy
    # Start with each leaf as its own set
    # For each split (in order of increasing size), merge the two parts
    
    # Actually, the cleanest way: 
    # The splits define a tree. We can build it by:
    # 1. Create a node for each split
    # 2. Connect split nodes that are "nested"
    # 3. Connect leaves to the smallest split that contains them
    
    # For simplicity, let's just build a Newick string directly
    # by recursively partitioning.
    
    def build_tree(leaves):
        """Build a subtree for the given set of leaves."""
        leaves = frozenset(leaves)
        if len(leaves) == 1:
            return f"t{list(leaves)[0]}"
        if len(leaves) == 2:
            a, b = sorted(leaves)
            return f"(t{a},t{b})"
        
        # Find the smallest split that partitions these leaves
        # A split s partitions leaves if s & leaves is non-empty and (leaves - s) is non-empty
        # and s is "minimal" in the sense that no other split in our set
        # is a proper subset of s & leaves
        
        # Find all splits that intersect leaves
        candidate_splits = []
        for s in splits:
            left = s & leaves
            right = leaves - s
            if len(left) > 0 and len(right) > 0:
                candidate_splits.append((left, right))
        
        if not candidate_splits:
            # Star topology for these leaves
            return "(" + ",".join(f"t{i}" for i in sorted(leaves)) + ")"
        
        # Pick the split that gives the most balanced partition
        # or the smallest left side
        best_split = min(candidate_splits, key=lambda x: min(len(x[0]), len(x[1])))
        left, right = best_split
        
        left_tree = build_tree(left)
        right_tree = build_tree(right)
        return f"({left_tree},{right_tree})"
    
    return build_tree(set(range(n_taxa)))


def reconstruct(sequences):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = _START_TIME + _DEADLINE - 10.0  # 10s safety margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Compute distance matrix
    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[(i, j)] = d
    
    # For small n, try exhaustive search
    if n <= 7:
        result = _exhaustive_best_topology(dist, n, deadline)
        if result is not None:
            # Convert indices back to labels
            # The result uses t0..t(n-1) which correspond to labels[0]..labels[n-1]
            # Since labels are sorted, t0=labels[0], etc.
            # So the Newick is already correct
            return result + ";"
    
    # For larger n, use NJ
    newick = _nj_topology(dist)
    return newick + ";"
