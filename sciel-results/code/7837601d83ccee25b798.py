"""Rewritten program: CSP-style simultaneous row growth with gap representation,
beam search over partial assignments, then local search to balance spans."""
import random
import time
import heapq
from itertools import combinations

# EVOLVE-BLOCK-START

def _row_diffs(marks):
    """All positive pairwise differences of a row's marks."""
    d = set()
    n = len(marks)
    for i in range(n):
        for j in range(i + 1, n):
            d.add(marks[j] - marks[i])
    return d

def _greedy_row(used, k):
    """Greedy: build one row of k+1 marks starting at 0, smallest possible,
    avoiding used differences."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used):
            row.append(x)
            used |= new
    return row

def _greedy_baseline(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        used_local = set(used)
        row = _greedy_row(used_local, k)
        rows.append(row)
        used |= _row_diffs(row)
    return rows

def _row_from_gaps(gaps):
    marks = [0]
    s = 0
    for g in gaps:
        s += g
        marks.append(s)
    return marks

def _diffs_from_gaps(gaps):
    marks = _row_from_gaps(gaps)
    return _row_diffs(marks)

def _construct_beam(n_rows, k, deadline):
    """Beam search: each candidate is a list of rows, each row is a list of gaps
    (partial). We grow rows in a round-robin / most-constrained order.
    State: list of gap-lists (one per row). Objective: minimize max span,
    while keeping all diffs distinct."""
    best = None
    best_score = float('inf')
    beam = [[] for _ in range(n_rows)]  # each row's gaps so far
    used = set()
    # We'll grow each row to k gaps. Track progress.
    # Represent a state by the tuple of gap-tuples + used set (hashable via frozenset)
    # For beam, we keep a list of (score, state) where state = (gaps_list, used_frozen, spans)
    # To keep it manageable, we grow one row at a time, choosing the row with
    # the largest current span / most "urgent" to extend, or the smallest.

    # Actually, let's do a simpler beam: maintain a set of partial solutions.
    # Each partial solution: rows_gaps = [tuple of gaps for row i], used = set, 
    # and we track which rows are complete (len == k).
    # We grow by picking a row that is not complete, trying candidate next gaps.

    start = time.time()
    # Initial: all rows empty
    states = [([], set(), [0]*n_rows)]  # (list of gap-lists, used set, spans)
    # We'll do a best-first / beam: at each step, expand the most promising state
    # by adding one gap to one row.

    # To keep beam width, we limit states.
    beam_width = 200
    max_states = 20000

    # Priority: minimize max span, then minimize number of incomplete rows
    def state_priority(state):
        gaps_list, used, spans = state
        max_span = max(spans) if spans else 0
        incomplete = sum(1 for g in gaps_list if len(g) < k)
        return (max_span, incomplete)

    heap = []
    initial = ([[] for _ in range(n_rows)], set(), [0]*n_rows)
    heapq.heappush(heap, (state_priority(initial), 0, initial))
    seen = {frozenset((tuple(g) for g in initial[0]))}  # track gap patterns to avoid dup

    while heap and time.time() < deadline:
        pr, _, state = heapq.heappop(heap)
        gaps_list, used, spans = state
        # Check if complete
        if all(len(g) == k for g in gaps_list):
            max_span = max(spans)
            if max_span < best_score:
                best_score = max_span
                best = [list(g) for g in gaps_list]
            continue
        # Find rows that can be extended
        # For each incomplete row, try candidate next gaps
        # To limit branching, only try a few candidates per row
        candidates = []
        for r in range(n_rows):
            if len(gaps_list[r]) >= k:
                continue
            row_marks = _row_from_gaps(gaps_list[r])
            # Try next gap values
            # The next mark must be > last mark
            last = row_marks[-1]
            # Try gap from 1 up to some limit
            # Limit: we don't want to try too many; cap at, say, 15 per row
            max_try = 12
            for gap in range(1, max_try + 1):
                new_mark = last + gap
                # Check new diffs
                new_diffs = {new_mark - a for a in row_marks}
                if new_diffs & used:
                    continue
                # Also check within-row: new_mark - a for a in row_marks are all distinct
                # (they are, since row_marks are distinct and new_mark is new)
                candidates.append((r, gap, new_diffs, new_mark))
        # If no candidates, this state is a dead end
        if not candidates:
            continue
        # Expand: for each candidate, create a new state
        # To limit beam, we only expand the top few candidates
        # Score the candidate by the resulting max span
        scored = []
        for (r, gap, new_diffs, new_mark) in candidates:
            new_gaps_list = [list(g) for g in gaps_list]
            new_gaps_list[r].append(gap)
            new_spans = list(spans)
            new_spans[r] = new_mark
            new_used = used | new_diffs
            new_state = (new_gaps_list, new_used, new_spans)
            pr2 = state_priority(new_state)
            scored.append((pr2, new_state))
        # Sort by priority and keep top few
        scored.sort(key=lambda x: x[0])
        # Keep top 5 expansions per state to limit branching
        for pr2, new_state in scored[:5]:
            key = frozenset((tuple(g) for g in new_state[0]))
            if key in seen:
                continue
            seen.add(key)
            heapq.heappush(heap, (pr2, 0, new_state))
            if len(heap) > max_states:
                # Prune: pop the worst
                heapq.heappop(heap)

    if best is not None:
        return [_row_from_gaps(g) for g in best]
    return None

def _local_search(rows, n_rows, k, deadline):
    """Local search: swap gaps between rows to reduce max span."""
    # Convert to gaps
    gaps_list = []
    for row in rows:
        gaps = [row[i+1] - row[i] for i in range(k)]
        gaps_list.append(gaps)
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    best_rows = rows
    best_span = max(max(r) for r in rows)
    start = time.time()
    # Random swaps
    rng = random.Random(42)
    while time.time() < deadline:
        # Pick two rows
        r1, r2 = rng.sample(range(n_rows), 2)
        # Pick a position in each
        p1 = rng.randint(0, k - 1)
        p2 = rng.randint(0, k - 1)
        g1, g2 = gaps_list[r1][p1], gaps_list[r2][p2]
        if g1 == g2:
            continue
        # Swap
        gaps_list[r1][p1], gaps_list[r2][p2] = g2, g1
        # Check validity
        new_used = set()
        valid = True
        new_rows = []
        for r in range(n_rows):
            marks = _row_from_gaps(gaps_list[r])
            d = _row_diffs(marks)
            if d & new_used:
                valid = False
                break
            new_used |= d
            new_rows.append(marks)
        if not valid:
            # Revert
            gaps_list[r1][p1], gaps_list[r2][p2] = g1, g2
            continue
        # Check if any gap is 0 (invalid)
        if any(g <= 0 for row_gaps in gaps_list for g in row_gaps):
            gaps_list[r1][p1], gaps_list[r2][p2] = g1, g2
            continue
        span = max(max(r) for r in new_rows)
        if span < best_span:
            best_span = span
            best_rows = new_rows
        # Revert (we only keep if it improves, but for search we revert to explore)
        gaps_list[r1][p1], gaps_list[r2][p2] = g1, g2
    return best_rows

def construct(n_rows, k):
    start = time.time()
    deadline = start + 120  # 120s wall-clock limit, keep margin
    # First, get a baseline
    baseline = _greedy_baseline(n_rows, k)
    best_rows = baseline
    best_span = max(max(r) for r in baseline)
    # Try beam search with remaining time
    remaining = deadline - time.time()
    if remaining > 5:
        beam_result = _construct_beam(n_rows, k, time.time() + remaining - 10)
        if beam_result is not None:
            span = max(max(r) for r in beam_result)
            if span < best_span:
                best_rows = beam_result
                best_span = span
    # Local search on the best
    remaining = deadline - time.time()
    if remaining > 5:
        ls_result = _local_search(best_rows, n_rows, k, time.time() + remaining - 5)
        span = max(max(r) for r in ls_result)
        if span < best_span:
            best_rows = ls_result
            best_span = span
    return best_rows

# EVOLVE-BLOCK-END
