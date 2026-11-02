<<<<<<< SEARCH
"""Difference Triangle Set (4,7) solver.

Combined strategy:
- Fast greedy baseline + many randomized greedy variants (with row-order shuffling).
- Efficient local compactation (right-to-left, set-based difference checks).
- Time-bounded simulated annealing with:
    * single-element decrease (compaction) moves
    * random element perturbations + recompact
    * full-row rebuilds (multiple attempts, bounded by current max)
    * full recompact passes
    * pair-swap moves between rows
- Safety margin before the 150 s wall-clock limit.
"""
import time
import random
import numpy as np
=======
"""Difference Triangle Set (4,7) solver.

Memetic algorithm:
- Generate diverse greedy solutions with varied parameters.
- Combine solutions via row-level crossover (take the row with smaller max).
- Aggressive multi-pass compaction after each crossover.
- Track best solution found.
- Safety margin before the 150 s wall-clock limit.
"""
import time
import random
import numpy as np
>>>>>>> REPLACE

