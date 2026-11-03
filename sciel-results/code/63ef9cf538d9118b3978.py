<<<<<<< SEARCH
"""
Program to find a rank-minimal exact tensor decomposition for multiplying
a 3 by 5 matrix by a 5 by 6 matrix.

Uses a block decomposition strategy:
- Split dimensions to create sub-blocks
- Apply Strassen's algorithm (rank 7) to 2x2x2 sub-blocks where possible
- Use schoolbook for remaining blocks
- Total rank: 88 (improvement over schoolbook rank 90)
"""
import time
import numpy as np
=======
"""
Program to find a rank-minimal exact tensor decomposition for multiplying
a 3 by 5 matrix by a 5 by 6 matrix.

Uses a block decomposition strategy:
- Split n=3 into [2,1], m=5 into [2,2,1], p=6 into [2,2,2]
- Apply Strassen's algorithm (rank 7) to 2x2x2 sub-blocks
- Use schoolbook for remaining sub-blocks
- Total rank: 84 (improvement over schoolbook rank 90)
"""
import time
import numpy as np
>>>>>>> REPLACE

