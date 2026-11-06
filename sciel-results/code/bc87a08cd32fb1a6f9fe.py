"""Sparse pancyclic graphs on 40 vertices.

We search for a sparse graph on 40 vertices that contains a simple cycle of
every length 3..40, minimizing (edge_count - 40).

Approach:
- Start from a base cycle C = (0, 1, 2, ..., 39, 0) which already gives a
  Hamiltonian cycle (length 40).
- To get cycles of lengths 3..39, we add "chords" (extra edges).
- A chord (u, v) where u and v are on the base cycle creates a new cycle
  of length (dist(u,v) + 1) where dist is the shorter arc length? No, actually
  adding chord (u,v) to the base cycle creates two cycles: one of length
  (d+1) and one of length (40-d+1) where d is the distance along the cycle
  between u and v. Wait, if we have a base cycle of length 40, and we add
  edge (u,v), then we get two cycles: one going u -> ... -> v along the
  shorter arc + the chord, and one going u -> ... -> v along the longer arc
  + the chord. The lengths are (d+1) and (40-d+1) where d is the number of
  edges on the shorter arc. Actually, let d be the distance (number of edges)
  between u and v along the cycle. Then the two cycles have lengths d+1 and
  (40-d)+1 = 41-d. But we only need cycles of lengths 3..39. The base cycle
  already gives length 40. So we need to cover lengths 3..39.

Actually, let me reconsider. The base cycle is 0-1-2-...-39-0. This is a
cycle of length 40. If we add a chord between vertex i and vertex j, we
create two new cycles. One cycle goes from i to j along the path i, i+1, ..., j
(with wraparound) and back via the chord. The length of this cycle is (j-i mod 40) + 1.
The other cycle goes the other way, with length (40 - (j-i mod 40)) + 1.

So each chord (i, j) covers two cycle lengths: a = (j-i) % 40 + 1 and b = 41 - a,
where we take a in range [1, 40]. But if a = 1, that's a self-loop which isn't
allowed. If a = 40, that's the base cycle itself. So valid chords give a in [2, 39]
or a in [1, 39] but a=1 means i=j which is invalid. So a is in [2, 39] and b = 41-a
is also in [2, 39]. Wait, if a=2, b=39. If a=39, b=2. If a=20, b=21. So each
chord covers exactly two lengths from the set {2, 3, ..., 39}, but we don't need
length 2 (no 2-cycles in simple graphs). Actually length 2 would be a multi-edge
which isn't allowed. So we need to cover lengths 3..39.

Hmm, but a chord can also create cycles that are not just the two "obvious" ones
if combined with other chords. But for a simpler approach, let's consider that
each chord primarily covers the two cycle lengths mentioned. We need to select
a minimum set of chords such that every length in 3..39 is covered by at least
one chord's pair of lengths.

This is a set cover problem. Let's enumerate all possible chords and the lengths
they cover, then find a minimum set cover.

But wait, the problem says "every required edge must exist" for the cycles. So
the cycles we report must use only edges that are in the graph. If we just add
chords to the base cycle, the cycles of length d+1 and 41-d are indeed simple
cycles using the base cycle edges and the chord. So that works.

Let's build the set cover:
- Universe: {3, 4, ..., 39}
- For each pair (i, j) with i < j, compute d = (j - i) % 40. The two lengths
  are d+1 and 41-d. We only care about lengths in [3, 39].
  - If d+1 >= 3 and d+1 <= 39, it covers d+1.
  - If 41-d >= 3 and 41-d <= 39, it covers 41-d.
  - Note: d is in [1, 39] since i != j. d+1 is in [2, 40]. 41-d is in [2, 40].
  - We exclude length 2 and 40 (40 is already covered by base cycle).
  
Let's just compute for each chord the set of lengths in [3, 39] it covers.

Then we need a minimum set cover. Since the universe size is 37 (lengths 3..39),
and the number of chords is C(40,2) = 780, we can use a greedy set cover or
an exact algorithm. Given the small universe, a greedy approach might be
sufficient, but let's try to be smart.

Actually, we can observe that chords with d+1 = k cover length k, and chords with
41-d = k cover length k. So for each length k in [3, 39], we need at least one
chord that covers it. A chord covers k if d+1 = k (i.e., d = k-1) or 41-d = k
(i.e., d = 41-k). So for each k, the chords that cover it are those with distance
k-1 or 41-k.

This is a covering problem. Let's use a simple greedy set cover: repeatedly pick
the chord that covers the most uncovered lengths.

After selecting chords, we build the graph with the base cycle edges plus the
selected chords. Then we need to produce the actual cycles. For each length k in
3..39, we find a chord that covers k and construct the corresponding cycle.

Let's implement this.
"""

import itertools

N = 40

# EVOLVE-BLOCK-START
def construct():
    # Base cycle: 0-1-2-...-39-0
    base_edges = set()
    for i in range(N):
        base_edges.add((min(i, (i+1) % N), max(i, (i+1) % N)))
    
    # We need to cover lengths 3..39
    target_lengths = set(range(3, N))  # 3..39
    
    # Generate all chords and the lengths they cover
    chord_info = []  # list of (chord, covered_lengths_set)
    for i in range(N):
        for j in range(i+1, N):
            d = (j - i) % N
            # The two cycle lengths created
            len1 = d + 1
            len2 = N - d + 1
            covered = set()
            if 3 <= len1 <= 39:
                covered.add(len1)
            if 3 <= len2 <= 39:
                covered.add(len2)
            if covered:
                chord_info.append(((i, j), covered))
    
    # Greedy set cover
    uncovered = set(target_lengths)
    selected_chords = []
    
    while uncovered:
        # Find chord that covers most uncovered lengths
        best_chord = None
        best_cover = None
        best_count = -1
        for chord, covered in chord_info:
            if chord in [c for c, _ in selected_chords]:
                continue
            new_cover = covered & uncovered
            if len(new_cover) > best_count:
                best_count = len(new_cover)
                best_chord = chord
                best_cover = new_cover
        if best_count <= 0:
            break  # Should not happen
        selected_chords.append((best_chord, best_cover))
        uncovered -= best_cover
    
    # Build edge set
    edges = set(base_edges)
    for chord, _ in selected_chords:
        i, j = chord
        edges.add((min(i, j), max(i, j)))
    
    # Now construct cycles for each length 3..39
    # For each length k, find a selected chord that covers k, and build the cycle
    cycles = []
    # Map from length to chord that covers it
    length_to_chord = {}
    for chord, covered in selected_chords:
        for k in covered:
            if k not in length_to_chord:
                length_to_chord[k] = chord
    
    for k in range(3, N+1):
        if k == N:
            # Hamiltonian cycle: the base cycle
            cycle = list(range(N))
        else:
            chord = length_to_chord.get(k)
            if chord is None:
                # Fallback: try to find any chord that covers k
                for c, cov in chord_info:
                    if k in cov:
                        chord = c
                        break
            i, j = chord
            d = (j - i) % N
            len1 = d + 1
            len2 = N - d + 1
            if k == len1:
                # Cycle goes i -> i+1 -> ... -> j -> i
                cycle = []
                v = i
                for _ in range(d):
                    cycle.append(v)
                    v = (v + 1) % N
                cycle.append(j)
            else:
                # k == len2, cycle goes i -> i-1 -> ... -> j -> i
                cycle = []
                v = i
                for _ in range(N - d):
                    cycle.append(v)
                    v = (v - 1) % N
                cycle.append(j)
        cycles.append(cycle)
    
    return {'edges': list(edges), 'cycles': cycles}
# EVOLVE-BLOCK-END
