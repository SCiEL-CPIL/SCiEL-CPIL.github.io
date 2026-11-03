"""Strassen-based block decomposition for 3x4 by 4x7 matrix multiplication."""
import time
import sys

DIMENSIONS = [3, 4, 7]

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 140.0  # 10s safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication.
    
    Uses Strassen's algorithm on 2x2 blocks where possible, combined with schoolbook
    for the remaining parts.
    
    The key idea: partition the 3x4 matrix A and 4x7 matrix B into blocks,
    use Strassen for 2x2 block multiplications, and schoolbook for the rest.
    
    For A (3x4) * B (4x7) -> C (3x7):
    
    Let's partition A as:
    A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x2, A21 is 1x2, A22 is 1x2
    
    B = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x2, B21 is 2x2, B22 is 2x2
    
    C = A*B = [C11 C12; C21 C22] where C11 is 2x2, C12 is 2x2, C21 is 1x2, C22 is 1x2
    
    For the top-left 2x2 block: C11 = A11*B11 + A12*B21
    For the top-right 2x2 block: C12 = A11*B12 + A12*B22
    For the bottom-left 1x2 block: C21 = A21*B11 + A22*B21
    For the bottom-right 1x2 block: C22 = A21*B12 + A22*B22
    
    Each 2x2 * 2x2 multiplication can use Strassen (rank 7 instead of 8).
    Each 1x2 * 2x2 multiplication is just 4 scalar multiplications (rank 4).
    
    Total rank: 4 * 7 (Strassen for 2x2*2x2) + 4 * 4 (schoolbook for 1x2*2x2) = 28 + 16 = 44
    
    Wait, let me reconsider. We have:
    - C11 = A11*B11 + A12*B21: two 2x2*2x2 products -> 2*7 = 14 rank
    - C12 = A11*B12 + A12*B22: two 2x2*2x2 products -> 2*7 = 14 rank
    - C21 = A21*B11 + A22*B21: two 1x2*2x2 products -> 2*4 = 8 rank
    - C22 = A21*B12 + A22*B22: two 1x2*2x2 products -> 2*4 = 8 rank
    
    Total: 14 + 14 + 8 + 8 = 44
    
    But wait - we can be smarter. Let me think about this differently.
    
    Actually, let me use a different partition. Let's split the 4-dimensional middle
    into 2+2, and handle the 3 and 7 dimensions carefully.
    
    A is 3x4, B is 4x7.
    
    Partition A's rows into 2+1, and B's columns into 5+2 (or other splits).
    
    Let me try: A rows split as 2+1, B cols split as 3+4 (or 2+5, etc.)
    
    Actually, the cleanest approach: split the 4 middle dimension into 2+2.
    
    A = [A_top; A_bottom] where A_top is 2x4, A_bottom is 1x4
    B = [B_left; B_right] where B_left is 4x3, B_right is 4x4
    
    C = A_top * B + A_bottom * B  (3x7 result)
    
    C_top = A_top * B  (2x7)
    C_bottom = A_bottom * B  (1x7)
    
    For A_top * B: A_top is 2x4, B is 4x7.
    Split A_top's 4 cols into 2+2: A_top = [A_top_left | A_top_right] (each 2x2)
    Split B's 4 rows into 2+2: B = [B_top; B_bottom] (each 4x... wait B is 4x7)
    
    Hmm, B is 4x7. If I split its 4 rows into 2+2: B = [B_top; B_bottom] where B_top is 2x7, B_bottom is 2x7.
    
    A_top * B = A_top_left * B_top + A_top_right * B_bottom  (2x7)
    
    A_top_left is 2x2, B_top is 2x7.
    A_top_right is 2x2, B_bottom is 2x7.
    
    For 2x2 * 2x7: split B_top into 2x2 + 2x2 + 2x3 (cols 2+2+3)
    Actually, let me split 7 into 2+2+3.
    
    This is getting complicated. Let me just implement the straightforward Strassen-on-2x2-blocks approach.
    
    Simplest effective approach:
    - Split the 4 middle dimension into 2+2
    - A (3x4) = [A1; A2] where A1 is 2x4 (rows 0-1), A2 is 1x4 (row 2)
    - B (4x7) = [B1; B2] where B1 is 2x7 (rows 0-1), B2 is 2x7 (rows 2-3)
    
    C = A1*B1 + A2*B2  (3x7)
    
    A1*B1: 2x4 * 2x7 -> wait no. A1 is 2x4, B1 is 2x7. That doesn't multiply directly.
    
    Let me re-index. A is 3x4, B is 4x7.
    A[i,j] for i in 0..2, j in 0..3
    B[j,k] for j in 0..3, k in 0..6
    C[i,k] = sum_j A[i,j]*B[j,k]
    
    Split j into {0,1} and {2,3}:
    C[i,k] = sum_{j=0,1} A[i,j]*B[j,k] + sum_{j=2,3} A[i,j]*B[j,k]
    
    Let A_left[i,j] = A[i,j] for j in {0,1} (3x2)
    A_right[i,j] = A[i,j] for j in {2,3} (3x2)
    B_top[j,k] = B[j,k] for j in {0,1} (2x7)
    B_bot[j,k] = B[j,k] for j in {2,3} (2x7)
    
    C = A_left * B_top + A_right * B_bot  (3x7)
    
    A_left * B_top: 3x2 * 2x7 = 3x7
    A_right * B_bot: 3x2 * 2x7 = 3x7
    
    For 3x2 * 2x7:
    Split the 3 rows of A_left into 2+1:
    A_left = [AL_top; AL_bot] where AL_top is 2x2, AL_bot is 1x2
    
    AL_top * B_top: 2x2 * 2x7
    AL_bot * B_top: 1x2 * 2x7
    
    For 2x2 * 2x7: split 7 cols into 2+2+3
    = (2x2)*(2x2) + (2x2)*(2x2) + (2x2)*(2x3)
    = Strassen(7) + Strassen(7) + schoolbook(2*2*3=12)... 
    
    Hmm, 2x2 * 2x3: that's 4*3 = 12 scalar mults, rank 12.
    
    So 2x2 * 2x7 = 7 + 7 + 12 = 26 rank? That's worse than schoolbook 2*2*7=28.
    
    Let me try splitting 7 into 2+5:
    2x2 * 2x7 = (2x2)*(2x2) + (2x2)*(2x5) = 7 + 20 = 27. Still worse.
    
    Split 7 into 4+3:
    2x2 * 2x7 = (2x2)*(2x4) + (2x2)*(2x3)
    2x2 * 2x4 = (2x2)*(2x2) + (2x2)*(2x2) = 7+7 = 14
    2x2 * 2x3 = 12
    Total = 14 + 12 = 26. Still worse than 28.
    
    What about 7 = 2+2+2+1:
    2x2 * 2x7 = 3*(2x2*2x2) + (2x2*2x1) = 3*7 + 4 = 25. Better than 28!
    
    Or 7 = 2+2+3:
    2x2 * 2x7 = 2*7 + 12 = 26.
    
    Or 7 = 2+5:
    2x2 * 2x7 = 7 + 2*2*5 = 7+20 = 27.
    
    Best so far for 2x2*2x7 is 25 (using 2+2+2+1).
    
    Now for the full problem:
    C = A_left * B_top + A_right * B_bot
    
    A_left * B_top (3x2 * 2x7):
    Split 3 rows as 2+1:
    = (2x2 * 2x7) + (1x2 * 2x7)
    = 25 + 14 = 39
    
    A_right * B_bot (3x2 * 2x7):
    = 25 + 14 = 39
    
    Total: 39 + 39 = 78.
    
    Original schoolbook: 3*4*7 = 84.
    Improvement: 84 -> 78. Score = 84/78 ≈ 1.077.
    
    Can we do better? Let me try different splits.
    
    What if I split the 4 middle as 1+3?
    A_left is 3x1, A_right is 3x3
    B_top is 1x7, B_bot is 3x7
    
    A_left * B_top: 3x1 * 1x7 = 21 scalar mults (rank 21)
    A_right * B_bot: 3x3 * 3x7
    
    3x3 * 3x7: split 7 as 2+2+2+1
    3x3 * 3x2 = 18 scalar mults... or use Strassen? 3x3 can't directly use 2x2 Strassen.
    
    Split 3x3 as (2x3 + 1x3) * (3x2 + 3x2 + 3x2 + 3x1)... this gets messy.
    
    Let me try 4 = 2+2 split but handle 3x2*2x7 differently.
    
    3x2 * 2x7: Can I use a different strategy?
    View it as: for each of the 7 output columns, compute 3x2 * 2x1 = 3 outputs, each requiring 2 mults. Total 7*2 = 14 mults. That's already minimal for this sub-problem (can't do better than 14 since each output needs at least... well, 3*2*1/3... no, it's just 2 mults per output column, 7 columns = 14).
    
    So 3x2*2x7 = 14 is optimal.
    
    And 2x2*2x7: minimum rank? The tensor is [2,2,7]. 
    Rank of [2,2,7]... By submultiplicativity, rank >= 2*2 = 4 (trivial). 
    Actually rank([2,2,n]) = 2n for n odd? No...
    rank([2,2,2]) = 7 (Strassen)
    rank([2,2,3]) = ? At most 2*2*3=12, but can we do better?
    rank([2,2,4]) = 14 (two Strassen blocks)
    rank([2,2,7]) <= 3*7 + 4 = 25 (three Strassen 2x2x2 + one 2x2x1)
    
    Actually, is there a better way for 2x2x7?
    We can write 7 = 2+2+2+1, giving 3*7+4 = 25.
    Or 7 = 3+4: rank([2,2,3]) + rank([2,2,4]).
    rank([2,2,4]) = 14 (two Strassen).
    rank([2,2,3]): at most 12, but can we do better? 
    For 2x2x3, I don't think Strassen applies directly. 12 is likely optimal or close.
    So 12+14 = 26 > 25.
    
    So 25 seems good for 2x2x7.
    
    Total with 4=2+2 split: 2*(25+14) = 78.
    
    Let me try 4=1+1+2:
    A split into 3x1, 3x1, 3x2
    B split into 1x7, 1x7, 2x7
    
    C = A1*B1 + A2*B2 + A3*B3
    A1*B1: 3x1*1x7 = 21
    A2*B2: 3x1*1x7 = 21
    A3*B3: 3x2*2x7 = 14
    
    Total: 21+21+14 = 56!
    
    Wait, that's much better! Let me verify.
    
    A is 3x4. Split columns as 1+1+2:
    A = [A1 | A2 | A3] where A1 is 3x1, A2 is 3x1, A3 is 3x2
    
    B is 4x7. Split rows as 1+1+2:
    B = [B1; B2; B3] where B1 is 1x7, B2 is 1x7, B3 is 2x7
    
    C = A1*B1 + A2*B2 + A3*B3  (3x7)
    
    A1*B1: 3x1 * 1x7 = 3*1*7 = 21 scalar products, rank 21
    A2*B2: 3x1 * 1x7 = 21, rank 21
    A3*B3: 3x2 * 2x7 = 3*2*7 = 42 scalar products... 
    
    Wait, 3x2 * 2x7: the result is 3x7. Each output C[i,k] = sum_{j=0,1} A3[i,j]*B3[j,k]. That's 2 mults per output, 21 outputs = 42 scalar mults. Rank 42.
    
    Total: 21 + 21 + 42 = 84. Same as schoolbook!
    
    Hmm, I made an error. 3x2*2x7 is NOT 14. It's 3*2*7 = 42.
    
    Let me recompute. The rank of a tensor [n,m,p] for schoolbook is n*m*p.
    [3,1,7] has rank 21.
    [3,2,7] has rank 42.
    
    So 4=1+1+2 gives 21+21+42 = 84. No improvement.
    
    The improvement comes from using Strassen on 2x2x2 blocks. Let me redo.
    
    With 4=2+2:
    C = A_left*B_top + A_right*B_bot
    A_left: 3x2, B_top: 2x7 -> tensor [3,2,7], schoolbook rank 42
    A_right: 3x2, B_bot: 2x7 -> tensor [3,2,7], schoolbook rank 42
    Total: 84.
    
    Now apply Strassen to the 2x2 parts:
    In A_left*B_top (3x2 * 2x7):
    Split the 3 rows of A_left into 2+1:
    AL_top: 2x2, AL_bot: 1x2
    AL_top * B_top: 2x2 * 2x7 -> tensor [2,2,7]
    AL_bot * B_top: 1x2 * 2x7 -> tensor [1,2,7], rank 14
    
    [2,2,7]: use 3 Strassen [2,2,2] + 1 schoolbook [2,2,1] = 3*7 + 4 = 25
    [1,2,7]: rank 14
    
    So [3,2,7] -> 25 + 14 = 39.
    
    Two of these: 39*2 = 78.
    
    Score: 84/78 = 1.0769...
    
    Can we improve further? What about splitting 7 differently?
    
    [2,2,7]: Can we do better than 25?
    7 = 2+2+2+1: 3*7+4 = 25
    7 = 3+4: rank([2,2,3])+rank([2,2,4])
    rank([2,2,4]) = 14 (two Strassen)
    rank([2,2,3]): Let's think. 2x2*2x3. 
    We can split the 3 as 2+1: (2x2*2x2)+(2x2*2x1) = 7+4 = 11.
    So rank([2,2,3]) <= 11.
    Then [2,2,7] <= 11+14 = 25. Same.
    
    7 = 1+6: rank([2,2,1])+rank([2,2,6]) = 4 + 3*7 = 4+21 = 25. Same.
    
    So 25 is likely optimal for [2,2,7].
    
    What about a different overall strategy?
    
    Split 4 as 1+3:
    A1: 3x1, A2: 3x3
    B1: 1x7, B2: 3x7
    
    A1*B1: [3,1,7] rank 21
    A2*B2: [3,3,7]
    
    [3,3,7]: Split 3x3 as (2x3+1x3) and 7 as 2+2+2+1?
    This is getting complex. Let me just implement the 78-rank solution.
    
    Actually, let me try yet another approach. What if I split the 4 as 2+2 AND use Strassen more cleverly?
    
    Alternative: split A's 3 rows as 2+1 everywhere, and B's 7 cols as 2+2+2+1.
    
    Let me just implement the straightforward 78-rank solution:
    - Split middle 4 into 2+2
    - For each 3x2*2x7 sub-problem, split 3 rows into 2+1
    - Use Strassen for the 2x2*2x7 part (3 Strassen 2x2x2 + 1 schoolbook 2x2x1)
    - Schoolbook for the 1x2*2x7 part
    
    Let me code this up.
    
    Strassen for 2x2 * 2x2:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1+M4-M5+M7
    C01 = M3+M5
    C10 = M2+M4
    C11 = M1-M2+M3+M6
    
    For the tensor decomposition, each Strassen multiplication contributes 7 rank-1 terms.
    
    Let me build the full decomposition.
    """
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # We'll build U, V, W matrices
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (28, rank)  
    # W: (p*n, rank) = (21, rank)
    
    # Index conventions:
    # U[i*m + j, c] where i in 0..n-1, j in 0..m-1
    # V[j*p + k, c] where j in 0..m-1, k in 0..p-1
    # W[k*n + i, c] where k in 0..p-1, i in 0..n-1
    # 
    # The tensor is T[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c
    # And T[i,j,k] should equal 1 if this is a valid (i,j,k) term in A*B
    
    # Actually, the matrix multiplication tensor T[i,j,k] = delta for the product.
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid i,j,k (it's the identity mapping)
    # Wait no. The tensor decomposition represents:
    # (A*B)[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
    # Hmm, that's not quite right either.
    
    # Let me re-read the convention. The standard way:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: T[i,j,k] where T[i,j,k] = 1 (the multiplication tensor)
    # Decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # where U is indexed by (i,j), V by (j,k), W by (i,k)
    
    # In the given indexing:
    # U has shape (n*m, rank), U[i*m+j, c] = U[i,j,c]
    # V has shape (m*p, rank), V[j*p+k, c] = V[j,k,c]  
    # W has shape (p*n, rank), W[k*n+i, c] = W[k,i,c]... 
    
    # Wait, W[k*n+i, c]. The output is (A*B)[i,k]. So W should be indexed by (i,k) or (k,i)?
    # W[k*n+i] suggests the index is k*n+i, so it's W[k,i] in (k,i) order.
    # But (A*B)[i,k] = sum_j A[i,j]*B[j,k], so the output index is (i,k).
    # If W is indexed as W[k*n+i], that's W[k,i] which would correspond to output[k,i]?
    
    # The task says: "The output factor follows the official trace convention W[k*n+i,r]"
    # So W[k*n+i, r] means the (k,i) entry of the output matrix.
    # But (A*B) has shape (n,p) = (3,7), with entries (A*B)[i,k] for i in 0..2, k in 0..6.
    # So W[k*n+i] with k in 0..6, i in 0..2 gives index k*3+i, range 0..20.
    # This means W is indexed by (k,i) where k is the column of output, i is the row.
    # So W[k,i] = coefficient for (A*B)[i,k].
    
    # The tensor equation: (A*B)[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
    # 
    # For this to equal sum_j A[i,j]*B[j,k] for all A,B:
    # sum_c W[k*n+i,c] * U[i*m+j1,c] * V[j1*p+k,c] = delta[j1==j2]... 
    
    # Actually the condition is:
    # For all i,k: sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * A[i,j]) * (sum_j' V[j'*p+k,c] * B[j',k]) = sum_j A[i,j]*B[j,k]
    
    # This must hold for all A,B, so:
    # sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = 1 if j==j' and the (i,k) term, else 0
    # More precisely: sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = delta_{j,j} for matching...
    
    # The correct condition: the bilinear form must reproduce matrix multiplication.
    # sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for all i,j,k (each (i,j,k) triple)
    # Wait no, that's not right either. Let me think again.
    
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # Our formula: (A*B)[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] A[i,j]) * (sum_j' V[j'*p+k,c] B[j',k])
    # = sum_c sum_j sum_j' W[k*n+i,c] U[i*m+j,c] V[j'*p+k,c] A[i,j] B[j',k]
    
    # For this to equal sum_j A[i,j] B[j,k] for ALL A,B:
    # sum_c W[k*n+i,c] U[i*m+j,c] V[j'*p+k,c] = 1 if j==j' else 0
    # i.e., = delta_{j,j'}
    
    # So the condition is: for all i, j, j', k:
    # sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta_{j,j'}
    
    # In the schoolbook decomposition, each (i,j,k) triple gets its own column c:
    # U[i*m+j,c] = 1, V[j*p+k,c] = 1, W[k*n+i,c] = 1
    # Then sum_c ... = 1*1*1 = 1 for j==j' (same j), 0 otherwise. Correct.
    
    # Now let me implement the Strassen-based decomposition.
    
    # Strategy: split m=4 into m1=2, m2=2.
    # A = [A1 | A2] where A1 is n x m1 (3x2), A2 is n x m2 (3x2)
    # B = [B1; B2] where B1 is m1 x p (2x7), B2 is m2 x p (2x7)
    # (A*B)[i,k] = (A1*B1)[i,k] + (A2*B2)[i,k]
    
    # For A1*B1 (3x2 * 2x7):
    # Split n=3 into n1=2, n2=1.
    # A1 = [A1_top; A1_bot] where A1_top is 2x2, A1_bot is 1x2
    # (A1*B1)[i,k] = (A1_top*B1)[i,k] + (A1_bot*B1)[i,k] for i in 0..1, plus (A1_bot*B1)[i,k] for i=2
    
    # A1_top*B1 (2x2 * 2x7): Use Strassen-style
    # Split p=7 into p1=2, p2=2, p3=2, p4=1
    # B1 = [B1_1 | B1_2 | B1_3 | B1_4] (2x2, 2x2, 2x2, 2x1)
    # A1_top*B1_1: 2x2*2x2 -> Strassen (rank 7)
    # A1_top*B1_2: 2x2*2x2 -> Strassen (rank 7)
    # A1_top*B1_3: 2x2*2x2 -> Strassen (rank 7)
    # A1_top*B1_4: 2x2*2x1 -> schoolbook (rank 4)
    
    # A1_bot*B1 (1x2 * 2x7): schoolbook (rank 14)
    
    # Similarly for A2*B2.
    
    # Total rank: 2 * (7+7+7+4+14) = 2 * 39 = 78
    
    def strassen_222(offset_u, offset_v, offset_w, i_base, j_base, k_base, U, V, W):
        """
        Add Strassen decomposition for a 2x2 * 2x2 block.
        i_base, j_base, k_base: starting indices for i, j, k
        offset_u, offset_v, offset_w: column offsets in U, V, W
        Returns the number of columns used (7).
        """
        col = 0
        # Strassen factors for 2x2 * 2x2:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # For tensor T[i,j,k] where i,j in {0,1}, k in {0,1}:
        # We need: sum_c W[k,i,c] * U[i,j,c] * V[j,k,c] = delta_{j,j'} for the sub-problem
        
        # Strassen decomposition of [2,2,2] tensor:
        # The 7 rank-1 terms (U, V, W) entries for (i,j), (j,k), (k,i):
        
        # Term 1: M1 = (A00+A11)*(B00+B11)
        # U contributions: U[0,0]+=1, U[1,1]+=1 (A00 and A11)
        # V contributions: V[0,0]+=1, V[1,1]+=1 (B00 and B11)
        # W: W[0,0]+=1, W[0,1]+=1, W[1,0]+=1, W[1,1]+=1 (C00,C01,C10,C11 all get M1)
        # C00 += M1, C01 += M1, C10 += M1, C11 += M1
        
        # Actually, let me use the standard Strassen tensor decomposition.
        # The 7 vectors (u_c, v_c, w_c) for c=1..7:
        
        # c=1: M1=(a00+a11)(b00+b11)
        #   u: a00=1, a11=1, others 0
        #   v: b00=1, b11=1, others 0
        #   w: c00=1, c01=1, c10=1, c11=1
        
        # c=2: M2=(a10+a11)b00
        #   u: a10=1, a11=1
        #   v: b00=1
        #   w: c10=1, c11=-1
        
        # c=3: M3=a00(b01-b11)
        #   u: a00=1
        #   v: b01=1, b11=-1
        #   w: c01=1, c11=1
        
        # c=4: M4=a11(b10-b00)
        #   u: a11=1
        #   v: b10=1, b00=-1
        #   w: c10=1, c00=-1
        
        # c=5: M5=(a00+a01)b11
        #   u: a00=1, a01=1
        #   v: b11=1
        #   w: c00=-1, c11=1
        
        # c=6: M6=(a10-a00)(b00+b01)
        #   u: a10=1, a00=-1
        #   v: b00=1, b01=1
        #   w: c11=1
        
        # c=7: M7=(a01-a11)(b10+b11)
        #   u: a01=1, a11=-1
        #   v: b10=1, b11=1
        #   w: c00=1
        
        # Let me verify: C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # W values (w[k*n+i] for the 2x2 sub-block, so w[0,0]=w[0], w[0,1]=w[1], w[1,0]=w[2], w[1,1]=w[3]):
        # Actually in our indexing, for the sub-block with i_base, k_base:
        # W[k_base+0, i_base+0] = C00 coefficient
        # W[k_base+0, i_base+1] = C10 coefficient (wait, W is W[k,i])
        
        # W[k*n+i]: for the 2x2 sub-block, k in {k_base, k_base+1}, i in {i_base, i_base+1}
        # W[k_base*n + i_base] = C[i_base, k_base] coeff = C00
        # W[k_base*n + i_base+1] = C[i_base+1, k_base] coeff = C10
        # W[(k_base+1)*n + i_base] = C[i_base, k_base+1] coeff = C01
        # W[(k_base+1)*n + i_base+1] = C[i_base+1, k_base+1] coeff = C11
        
        # So W indices for the sub-block:
        # C00 -> W[k_base*n + i_base]
        # C10 -> W[k_base*n + i_base+1]
        # C01 -> W[(k_base+1)*n + i_base]
        # C11 -> W[(k_base+1)*n + i_base+1]
        
        # U indices: U[i*m+j]
        # U[i_base*m + j_base] = A00
        # U[i_base*m + j_base+1] = A01
        # U[(i_base+1)*m + j_base] = A10
        # U[(i_base+1)*m + j_base+1] = A11
        
        # V indices: V[j*p+k]
        # V[j_base*p + k_base] = B00
        # V[j_base*p + k_base+1] = B01
        # V[(j_base+1)*p + k_base] = B10
        # V[(j_base+1)*p + k_base+1] = B11
        
        # Now the 7 terms:
        terms = [
            # (U entries, V entries, W entries) as (index, value) lists
            # c=1: M1
            ([(i_base*m+j_base, 1), ((i_base+1)*m+(j_base+1), 1)],
             [(j_base*p+k_base, 1), ((j_base+1)*p+(k_base+1), 1)],
             [(k_base*n+i_base, 1), (k_base*n+i_base+1, 1), ((k_base+1)*n+i_base, 1), ((k_base+1)*n+i_base+1, 1)]),
            # c=2: M2
            ([(i_base*m+j_base+1, 1), ((i_base+1)*m+(j_base+1), 1)],
             [(j_base*p+k_base, 1)],
             [(k_base*n+i_base+1, 1), ((k_base+1)*n+i_base+1, -1)]),
            # c=3: M3
            ([(i_base*m+j_base, 1)],
             [(j_base*p+k_base+1, 1), ((j_base+1)*p+(k_base+1), -1)],
             [((k_base+1)*n+i_base, 1), ((k_base+1)*n+i_base+1, 1)]),
            # c=4: M4
            ([((i_base+1)*m+(j_base+1), 1)],
             [((j_base+1)*p+k_base, 1), (j_base*p+k_base, -1)],
             [(k_base*n+i_base+1, 1), (k_base*n+i_base, -1)]),
            # c=5: M5
            ([(i_base*m+j_base, 1), (i_base*m+j_base+1, 1)],
             [((j_base+1)*p+(k_base+1), 1)],
             [(k_base*n+i_base, -1), ((k_base+1)*n+i_base+1, 1)]),
            # c=6: M6
            ([((i_base+1)*m+j_base, 1), (i_base*m+j_base, -1)],
             [(j_base*p+k_base, 1), (j_base*p+k_base+1, 1)],
             [((k_base+1)*n+i_base+1, 1)]),
            # c=7: M7
            ([(i_base*m+j_base+1, 1), ((i_base+1)*m+(j_base+1), -1)],
             [((j_base+1)*p+k_base, 1), ((j_base+1)*p+(k_base+1), 1)],
             [(k_base*n+i_base, 1)]),
        ]
        
        for u_entries, v_entries, w_entries in terms:
            for idx, val in u_entries:
                U[idx][offset_u + col] += val
            for idx, val in v_entries:
                V[idx][offset_v + col] += val
            for idx, val in w_entries:
                W[idx][offset_w + col] += val
            col += 1
        
        return col
    
    def schoolbook_block(i_base, j_base, k_base, n_sub, m_sub, p_sub, offset, U, V, W):
        """
        Add schoolbook decomposition for an n_sub x m_sub * m_sub x p_sub block.
        Returns number of columns used.
        """
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[(i_base+i)*m + (j_base+j)][offset + col] = 1
                    V[(j_base+j)*p + (k_base+k)][offset + col] = 1
                    W[(k_base+k)*n + (i_base+i)][offset + col] = 1
                    col += 1
        return col
    
    # Initialize matrices
    # We expect rank = 78
    rank = 78
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Split m=4 into m1=2 (j=0,1) and m2=2 (j=2,3)
    # Split n=3 into n1=2 (i=0,1) and n2=1 (i=2)
    # Split p=7 into p1=2 (k=0,1), p2=2 (k=2,3), p3=2 (k=4,5), p4=1 (k=6)
    
    # For each half of m (m1 and m2):
    for m_half in range(2):
        j_base = m_half * 2  # 0 or 2
        
        # Top part: i=0,1 (2x2 * 2x7)
        # Use Strassen for the three 2x2x2 blocks and schoolbook for 2x2x1
        for p_block in range(3):
            k_base = p_block * 2  # 0, 2, 4
            col += strassen_222(col, col, col, 0, j_base, k_base, U, V, W)
        
        # Last p block: k=6 (2x2 * 2x1) - schoolbook
        col += schoolbook_block(0, j_base, 6, 2, 2, 1, col, U, V, W)
        
        # Bottom part: i=2 (1x2 * 2x7) - schoolbook
        col += schoolbook_block(2, j_base, 0, 1, 2, 7, col, U, V, W)
    
    # col should be 78
    # Verify: 2 * (3*7 + 4 + 14) = 2 * (21 + 4 + 14) = 2 * 39 = 78
    
    # Convert to integers if all values are integer
    def to_int_if_possible(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if abs(val - round(val)) < 1e-10:
                    new_row.append(int(round(val)))
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    U = to_int_if_possible(U)
    V = to_int_if_possible(V)
    W = to_int_if_possible(W)
    
    return {"U": U, "V": V, "W": W}
