<<<<<<< SEARCH
    # Sub-problem 8: A22 B22 (2x3x3) -> Schoolbook
    schoolbook(2, 3, 3, 2, 2, 2)
=======
    # Sub-problem 8: A22 B22 (2x3x3) -> Strassen(2x2x2) + Schoolbook(2x1x3)
    # i in {2,3}, j in {2,3,4}, k in {2,3,4}
    # Split j: {2,3} and {4}
    strassen_2x2x2(2, 2, 2)  # i in {2,3}, j in {2,3}, k in {2,3}
    schoolbook(2, 1, 3, 2, 4, 2) # i in {2,3}, j in {4}, k in {2,3,4}
>>>>>>> REPLACE
