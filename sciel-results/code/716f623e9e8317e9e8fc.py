"""Program: rank-minimal exact tensor decomposition for 3x5 * 5x6 matrix multiplication.

Kronecker product decomposition:
  3x5x6 = 3x1x1 (rank 3) + 1x5x5 (rank 11) + 2x1x6 (rank 7)
  Total rank = 21

The key insight is that C[i,k] = sum_j A[i,j] B[j,k] can be split as:
  C[i,k] = A[i,0]*B[0,k] + sum_{j=1}^{4} A[i,j]*B[j,k] + A[i,4]*B[4,k]
         = (3x1x1 part)    + (1x5x5 part: i=0)  + (2x1x6 part: i=1,2, j=4)

Wait, let me reconsider. The tensor T[i,j,k]=1 for all i,j,k with C[i,k]=sum_j A[i,j]B[j,k].

Actually, the decomposition is:
  C[i,k] = sum_{j=0}^{4} A[i,j] B[j,k]

We can partition the j-index. Let's use:
- j=0: contributes A[i,0]*B[0,k] for all i,k -> this is a 3x1x6 sub-tensor (rank 3)
- j=1..4: contributes sum_{j=1}^{4} A[i,j]*B[j,k] -> this is 3x4x6

For the 3x4x6 part, we can further decompose using Kronecker structure.

Actually, let me think more carefully. The standard approach:

The multiplication tensor M_{n,m,p} has rank at most:
  rank(M_{n,m,p}) <= n * rank(M_{1,m,p})
  rank(M_{n,m,p}) <= p * rank(M_{n,m,1})
  rank(M_{n,m,p}) <= m * rank(M_{n,1,p})

But we can do better with Kronecker products:
  M_{n1+n2, m, p} = M_{n1,m,p} + M_{n2,m,p} (direct sum, rank = sum)

So rank(M_{3,5,6}) <= rank(M_{1,5,6}) + rank(M_{2,5,6})

And rank(M_{1,5,6}) <= rank(M_{1,5,5}) + rank(M_{1,1,1})  [split p=5+1... no, p=6=5+1]
  Wait, M_{1,5,6}: C[0,k] = sum_{j=0}^{4} A[0,j] B[j,k] for k=0..5
  This is a 1x5x6 tensor. We can split k: k=0..4 (5 terms) and k=5 (1 term)
  rank(M_{1,5,6}) <= rank(M_{1,5,5}) + rank(M_{1,5,1})
  rank(M_{1,5,1}) = 5 (each term A[0,j]*B[j,0] is rank 1)
  
  Actually rank(M_{1,m,p}) = m*p? No...
  M_{1,m,p}[0,j,k] = 1 for all j,k. This is the all-ones tensor of size 1xm x p.
  rank(M_{1,m,p}) = max(m,p)? No, it's the rank of the 3-tensor which is m*p since
  each (j,k) pair needs its own term... unless we can share.
  
  Actually for M_{1,m,p}: T[0,j,k] = 1. We need sum_t U[j,t]*V[k,t]*W[0,t] = 1.
  If we set W[0,t]=1 for all t, then we need sum_t U[j,t]*V[k,t] = 1 for all j,k.
  This is an outer product decomposition of the all-ones matrix of size m x p.
  The rank of the all-ones matrix is 1, so U[j,t]=1, V[k,t]=1 for t=0 only gives 1.
  So rank(M_{1,m,p}) = 1! Just use one term with all ones.

Similarly rank(M_{n,m,1}) = 1 and rank(M_{n,1,p}) = 1.

So:
  rank(M_{3,5,6}) <= rank(M_{1,5,6}) + rank(M_{2,5,6})
                   = 1 + rank(M_{2,5,6})
  
  rank(M_{2,5,6}) <= rank(M_{2,5,5}) + rank(M_{2,5,1})
                   = rank(M_{2,5,5}) + 1
  
  rank(M_{2,5,5}): We can use Strassen-like decompositions.
  M_{2,5,5}: C[i,k] = sum_{j=0}^{4} A[i,j]*B[j,k] for i=0,1; k=0..4.
  
  One approach: group j into pairs. j=0,1 use Strassen (rank 7 for 2x2x2), 
  j=2,3 use Strassen (rank 7), j=4 use rank 1.
  Total: 7+7+1 = 15.
  
  Or: j=0,1,2,3,4. Group as (0,1), (2,3), (4).
  Each 2x2x2 Strassen block: rank 7.
  Single: rank 1.
  Total: 15.
  
  Can we do better? What about grouping (0,1,2,3,4) differently?
  
  Actually, let's reconsider the whole decomposition:
  
  rank(M_{3,5,6}) <= rank(M_{1,5,6}) + rank(M_{2,5,6})
  
  For M_{2,5,6}:
    C[i,k] = sum_{j=0}^{4} A[i,j]*B[j,k], i=0,1; k=0..5.
    
    Split k: k=0..4 and k=5.
    rank(M_{2,5,6}) <= rank(M_{2,5,5}) + rank(M_{2,5,1}) = rank(M_{2,5,5}) + 1
    
  For M_{2,5,5}:
    C[i,k] = sum_{j=0}^{4} A[i,j]*B[j,k], i=0,1; k=0..4.
    
    Split j: (0,1), (2,3), (4).
    rank(M_{2,5,5}) <= rank(M_{2,2,5}) + rank(M_{2,2,5}) + rank(M_{2,1,5})
                     = 2*rank(M_{2,2,5}) + rank(M_{2,1,5})
    
    rank(M_{2,2,5}): C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k], i=0,1; k=0..4.
      Split k: (0,1), (2,3), (4).
      rank(M_{2,2,5}) <= rank(M_{2,2,2}) + rank(M_{2,2,2}) + rank(M_{2,2,1})
                       = 2*7 + 2 = 16? 
  
      Wait, rank(M_{2,2,1}) = 2 (each of the 2 i-indices needs a term, or actually:
      T[i,j,0] = 1 for i=0,1; j=0,1. 
      sum_t U[i,j,t]*V[0,t]*W[i,t] = 1.
      If V[0,t]=1, then sum_t U[i,j,t]*W[i,t] = 1 for all i,j.
      Set W[0,t]=1, W[1,t]=1, U[i,j,0]=1 for all i,j. That's rank 1!
      So rank(M_{2,2,1}) = 1.
      
      Similarly rank(M_{1,2,2}) = 1, rank(M_{2,1,2}) = 1.
      
      So rank(M_{2,2,5}) <= 2*7 + 1 = 15.
      
    rank(M_{2,1,5}): C[i,k] = A[i,0]*B[0,k], i=0,1; k=0..4.
      T[i,0,k] = 1. sum_t U[i,0,t]*V[0,k,t]*W[i,t] = 1.
      Set U[i,0,0]=1, V[0,k,0]=1, W[i,0]=1. Rank 1.
      
    So rank(M_{2,5,5}) <= 2*15 + 1 = 31.
    
  Total: 1 + 31 + 1 = 33. That's not great.

Let me try a different decomposition. The directive says:
  3x5x6 = 3x1x1 (rank 3) + 1x5x6 (rank 11) + 2x1x6 (rank 7) = 21

Hmm, that doesn't match my analysis. Let me re-read.

"expressing the 3x5x6 multiplication as a direct sum of a 3x1x1 component and a 1x5x6 component, then recursively decompose the 1x5x6 part using a rank-11 decomposition for 1x5x5 (derived from the 2x5x5 case via zero-padding) combined with a rank-7 Strassen block for the 2x1x6 remainder"

This is confusing. Let me just try to get a good rank using Strassen blocks.

Strategy: Use Strassen's algorithm recursively.
- M_{3,5,6}: Split into M_{1,5,6} + M_{2,5,6} (rank = sum)
- M_{1,5,6} = rank 1 (all ones tensor)
- M_{2,5,6}: Split k into 5+1: M_{2,5,5} + M_{2,5,1}
  - M_{2,5,1} = rank 1
  - M_{2,5,5}: Split j into 2+2+1: M_{2,2,5} + M_{2,2,5} + M_{2,1,5}
    - M_{2,1,5} = rank 1
    - M_{2,2,5}: Split k into 2+2+1: M_{2,2,2} + M_{2,2,2} + M_{2,2,1}
      - M_{2,2,2} = Strassen rank 7
      - M_{2,2,1} = rank 1
      - So M_{2,2,5} = 7+7+1 = 15
    - M_{2,5,5} = 15+15+1 = 31
  - M_{2,5,6} = 31+1 = 32
- Total = 1 + 32 = 33

Alternatively, use a different split. What if we use:
- M_{3,5,6}: Split i into 1+2.
- M_{1,5,6}: rank 1
- M_{2,5,6}: 
  - Split j into 2+2+1
  - M_{2,2,6}: Split k into 2+2+2: 3*M_{2,2,2} = 3*7 = 21
  - M_{2,2,6} = 21
  - M_{2,2,6} = 21  
  - M_{2,1,6} = rank 1
  - M_{2,5,6} = 21+21+1 = 43
- Total = 1+43 = 44. Worse.

What about:
- M_{3,5,6}: Split j into 2+2+1
- M_{3,2,6}: Split k into 2+2+2: 3*M_{3,2,2}
  - M_{3,2,2}: Split i into 1+2: M_{1,2,2} + M_{2,2,2} = 1+7 = 8
  - M_{3,2,6} = 3*8 = 24
- M_{3,2,6} = 24
- M_{3,1,6} = rank 1
- Total = 24+24+1 = 49. Worse.

Let me try:
- M_{3,5,6}: Split i into 1+1+1
- Each M_{1,5,6} = rank 1
- Total = 3. 

Wait! Is rank(M_{1,5,6}) really 1? Let me verify.
T[0,j,k] = 1 for all j in [0,4], k in [0,5].
We need: sum_t U[j,t] * V[k,t] * W[0,t] = 1 for all j,k.
Set t=0: U[j,0]=1, V[k,0]=1, W[0,0]=1. Then sum = 1*1*1 = 1. Yes!
So rank(M_{1,5,6}) = 1.

Similarly rank(M_{3,5,1}) = 1, rank(M_{3,1,6}) = 1.

So:
- M_{3,5,6} = M_{1,5,6} + M_{1,5,6} + M_{1,5,6} (split i=0,1,2)
- Each M_{1,5,6} has rank 1.
- Total rank = 3!

Let me verify this more carefully.

For the full tensor T[i,j,k] = 1 (i=0..2, j=0..4, k=0..5):
Decomposition with rank 3:
  U[i*m+j, t] = 1 if t == i, else 0  (U has shape 15x3)
  V[j*p+k, t] = 1 for all j,k,t     (V has shape 30x3)
  W[k*n+i, t] = 1 if t == i, else 0  (W has shape 18x3)

Check: sum_t U[i*5+j, t] * V[j*6+k, t] * W[k*3+i, t]
  = sum_t [t==i] * 1 * [t==i]
  = [i==i] = 1. ✓

So rank 3 works! The current program's `_try_rank_1` actually returns rank 1 which is WRONG (it doesn't verify correctly or the verification is broken).

Wait, looking at the current code again: `_try_rank_1` sets U, V, W all to all-ones with rank 1. That gives:
sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1*1*1 = 1. 

Oh wait, that IS correct! With rank 1, all entries are 1, so the product is always 1. Let me re-examine.

U[i*m+j, 0] = 1 for all i,j
V[j*p+k, 0] = 1 for all j,k  
W[k*n+i, 0] = 1 for all k,i

sum_t=0^0 U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1. ✓

So rank 1 should work! But the score is 1.0 which equals the seed. Let me check if the verification is failing.

Looking at `_verify`: it checks `total != 1`. With integer arithmetic this should pass. But maybe the issue is that the evaluator expects a specific format or the rank is being measured differently.

Actually wait - re-reading the task: "score is the rank". The current program returns rank 1 via `_try_rank_1` if verification passes. Score 1.0 means seed/value = 1.0, so value = seed. If the seed rank is 1, then score 1.0 means it's achieving rank 1.

But can rank be less than 1? No, minimum rank is 1 (or 0 if the tensor is zero, which it's not). So rank 1 is optimal and the score of 1.0 is already optimal!

Hmm, but the task says "Find a rank-minimal exact tensor decomposition". If rank 1 works, then 1 is minimal. The score being 1.0 (equal to seed) confirms this is already optimal.

Let me double-check by re-reading: "The evaluator measures the raw value (lower is better). The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement."

So if seed has rank R_seed and we achieve rank R, score = R_seed/R. For score > 1.0, we need R < R_seed. If R_seed = 1, we can't improve.

But wait, maybe the seed doesn't actually achieve rank 1. Let me look more carefully at whether the verification passes.

The `_verify` function uses `dtype=object` and checks `total != 1`. With all integer 1s, total = 1, so it should pass. Unless there's a bug.

Actually, I think the issue might be that the program IS achieving rank 1 and the score of 1.0 is already optimal (can't go below 1). In that case, no improvement is possible.

But let me reconsider - maybe the evaluator computes rank differently, or maybe there's a constraint I'm missing about the entries needing to be "integer/half-integer" in a specific way.

Given the analysis, rank 1 is provably optimal (minimum possible rank for a non-zero tensor). The current program already achieves this. No improvement is possible.

However, let me make sure the code is robust and the verification definitely passes. Let me clean it up slightly to ensure correctness.

